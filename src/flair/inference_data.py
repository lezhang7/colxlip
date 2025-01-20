import ast
import json
import logging
import math
import os
import re
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
import random

import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import collections

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def split_caption(text):
    """Split captions by sentence-ending markers."""
    return [cap.strip() for cap in re.split(r'\n|</s>|[.]', text) if cap.strip()]

def random_sample_from_list(captions_list, k, merged_num=1):
    n = len(captions_list)
    if merged_num == 1:
        if n >= k:
            return random.sample(captions_list, k)
        else:  #minimizing caption dupilications
            return random.choices(captions_list, k=k)
            #return captions_list + random.sample(captions_list, k - n)
    elif merged_num >= n:
        return ['. '.join(captions_list)]
    else:
        sampled_list = []
        sampled_indices = draw_numbers(n=n - merged_num, k=k)
        for sampled_index in sampled_indices:
            sampled_list.append('. '.join(captions_list[sampled_index:sampled_index + merged_num]))
        return sampled_list


def draw_numbers(n, k=4):
    population = list(range(0, n))
    if n >= k:
        return random.sample(population, k)
    else:
        return random.choices(population, k=k)




def draw_numbers(n, k=4):
    population = list(range(0, n))
    if n >= k:
        return random.sample(population, k)
    else:
        return random.choices(population, k=k)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def filter_no_caption_or_no_image_json(sample):
    has_caption = ('json' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()



def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.retrieval_coco:
        data["retrieval_coco"] = get_retrieval_coco_dataset(args=args, preprocess_fn=preprocess_val,
                                                            tokenizer=tokenizer, output_dict=True)
    if args.retrieval_flickr:
        data["retrieval_flickr"] = get_retrieval_flickr_dataset(args=args, preprocess_fn=preprocess_val,
                                                                tokenizer=tokenizer, output_dict=True)
    if args.retrieval_docci:
        data["retrieval_docci"] = get_finegrained_or_long_retrieval_dataset(args=args, preprocess_fn=preprocess_val,
                                                                    tokenizer=tokenizer, output_dict=True,
                                                                    dataset_name='docci')
    if args.retrieval_urban_1k:
        data["retrieval_urban_1k"] = get_finegrained_or_long_retrieval_dataset(args=args, preprocess_fn=preprocess_val,
                                                                       tokenizer=tokenizer, output_dict=True,
                                                                       dataset_name='urban-1k')

    if args.retrieval_dci:
        data["retrieval_dci"] = get_finegrained_or_long_retrieval_dataset(args=args, preprocess_fn=preprocess_val,
                                                                  tokenizer=tokenizer, output_dict=True,
                                                                  dataset_name='dci')

    if args.retrieval_iiw:
        data["retrieval_iiw"] = get_finegrained_or_long_retrieval_dataset(args=args, preprocess_fn=preprocess_val,
                                                                  tokenizer=tokenizer, output_dict=True,
                                                                  dataset_name='iiw')

    if args.retrieval_sharegpt4v_1k:
        data["retrieval_sharegpt4v-1k"] = get_finegrained_or_long_retrieval_dataset(args=args, preprocess_fn=preprocess_val,
                                                                            tokenizer=tokenizer, output_dict=True,
                                                                            dataset_name='sharegpt4v-1k')

    if args.retrieval_sharegpt4v_10k:
        data["retrieval_sharegpt4v-10k"] = get_finegrained_or_long_retrieval_dataset(args=args, preprocess_fn=preprocess_val,
                                                                             tokenizer=tokenizer, output_dict=True,
                                                                             dataset_name='sharegpt4v-10k')
    return data




def read_coco_pairs(root_dir, dict_root_dir, split='train', sampling_mode=None, num_samples=None):
    """
    :param num_samples: int
    :param sampling_mode: str.
    :param root_dir: str; path to the dataset folder
    :param split: str; 'train' or 'val'
    :return: a list of dict: {'image_id': int, 'image': str, 'caption': str}
    """
    annotations_dir = os.path.join(root_dir, "annotations")
    if split == "train":
        captions_file = os.path.join(annotations_dir, "captions_train2017.json")
        images_dir = os.path.join(root_dir, "images", "train2017")
    else:
        split = 'val'
        captions_file = os.path.join(annotations_dir, "captions_val2017.json")
        images_dir = os.path.join(root_dir, "images", "val2017")

    with open(captions_file, 'r') as f:
        coco_data = json.load(f)

    image_id_to_path = {image['id']: os.path.join(images_dir, image['file_name']) for image in coco_data['images']}
    data_list = []
    cap_id = 0
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id in image_id_to_path:
            data_list.append({
                'image_id': image_id,
                'image': image_id_to_path[image_id],
                'caption': annotation['caption'],
                'caption_id': cap_id
            })
        cap_id += 1

    return data_list


def map_img_cap(data_list):
    """
    :param data_list: List of dict, each dict contains key 'image_id' and 'caption_id'
    :return: img2txt_dict, txt2img_dict
    """
    img2txt_dict = {}
    txt2img_dict = {}

    for entry in data_list:
        image_id = entry['image_id']
        caption_id = entry['caption_id']

        if image_id not in img2txt_dict:
            img2txt_dict[image_id] = [caption_id]
        else:
            img2txt_dict[image_id].append(caption_id)

        if caption_id not in txt2img_dict:
            txt2img_dict[caption_id] = [image_id]
        else:
            txt2img_dict[caption_id].append(image_id)
    return img2txt_dict, txt2img_dict



def read_flickr_pairs(root_dir, split='train'):
    base_dir = os.path.dirname(root_dir)
    if split == 'train':
        captions_file = os.path.join(root_dir, "flickr30k_train.json")
    elif split == 'val':
        captions_file = os.path.join(root_dir, "flickr30k_val.json")
    else:
        captions_file = os.path.join(root_dir, "flickr30k_test.json")

    with open(captions_file, 'r') as f:
        flickr_data = json.load(f)

    data_list = []
    img_id, cap_id = 0, 0
    for annotation in flickr_data:
        image_path = os.path.join(base_dir, annotation['image'])
        caption_list = annotation["caption"]  # Now the caption should be a list
        for caption in caption_list:
            data_list.append({
                'image': image_path,
                'caption': caption,
                'image_id': img_id,
                'caption_id': cap_id
            })
            cap_id += 1
        img_id += 1
    return data_list


def read_cc3m_pairs(root_dir, split='train_retrieval'):
    if split == 'train_retrieval':
        captions_file = os.path.join(root_dir, "annotations", "annotations.json")
    else:
        raise NotImplementedError("the fine-grained retrieval on CC3M val is still TODO")

    with open(captions_file, 'r') as f:
        cc3m_data = json.load(f)['annotations']  #should be a list of dict

    data_list = []
    cap_id = 0
    for annotation in cc3m_data:
        image_path = os.path.join(root_dir, 'images', annotation['image'])
        data_list.append({
            'image': image_path,
            'caption': annotation['caption'],
            'image_id': annotation['image_id'],
            'caption_id': cap_id
        })
        cap_id += 1
    return data_list


def read_docci_pairs(root_dir, split='test'):
    if split == 'test':
        captions_file = os.path.join(root_dir, "annotations", "test_annotations.json")
    else:
        raise NotImplementedError("the fine-grained retrieval on other folds of DOCCI is still TODO")

    with open(captions_file, 'r') as f:
        docci_data = json.load(f)['annotations']  #should be a list of dict

    data_list = []
    cap_id = 0
    for annotation in docci_data:
        image_path = os.path.join(root_dir, 'images', annotation['image'])
        data_list.append({
            'image': image_path,
            'caption': annotation['caption'],
            'image_id': annotation['image_id'],
            'caption_id': cap_id
        })
        cap_id += 1
    return data_list


def read_urban1k_pairs(root_dir, split='test'):
    if split == 'test':
        captions_file = os.path.join(root_dir, "annotations", "annotations.json")
    else:
        raise NotImplementedError("There is only one retrieval mode for urban1k")

    with open(captions_file, 'r') as f:
        urban1k_data = json.load(f)['annotations']  #should be a list of dict

    data_list = []
    cap_id = 0
    for annotation in urban1k_data:
        image_path = os.path.join(root_dir, 'images', annotation['image'])
        data_list.append({
            'image': image_path,
            'caption': annotation['caption'],
            'image_id': annotation['image_id'],
            'caption_id': cap_id
        })
        cap_id += 1
    return data_list


def read_sharegpt4v_pairs(root_dir, json_name, total_len, split='test'):
    with open(json_name, 'r', encoding='utf8') as fp:
        json_data = json.loads(fp.read(), object_pairs_hook=collections.OrderedDict)[:total_len]
    fp.close()

    data_list = []
    for index in range(total_len):
        caption = json_data[index]['conversations'][1]['value']
        rejoined_caption = '. '.join(split_caption(caption))
        image_name = json_data[index]['image']
        if "images" in image_name:
            image_name = image_name.replace('/images', '')
        image_path = os.path.join(root_dir, image_name)
        data_list.append({
            'image': image_path,
            'caption': rejoined_caption,
            'image_id': index,
            'caption_id': index})
    return data_list


def read_dci_pairs(root_dir, split='test', splitted_captions=False):
    anno_file = os.path.join(root_dir, 'densely_captioned_images', 'splits.json')
    with open(anno_file, 'r', encoding='utf8') as fp:
        split = json.load(fp)
    data = []
    for k, v in split.items():
        data = data + v
    fp.close()
    #data is alist containing all image and captions pairs
    image_root = os.path.join(root_dir, 'densely_captioned_images', 'photos')
    anno_root = os.path.join(root_dir, 'densely_captioned_images', 'annotations')

    data_list = []
    cap_id = 0
    img_id = 0

    for data_file in data:
        with open(os.path.join(anno_root, data_file), 'r', encoding='utf8') as annotation_json:
            anno = json.load(annotation_json)
            image_path = os.path.join(image_root, anno['image'])
            caption = f"{anno['short_caption']}\n{anno['extra_caption']}"
            rejoined_caption = ". ".join(split_caption(caption))
            data_list.append({
                'image': image_path,
                'caption': rejoined_caption,
                'image_id': img_id,
                'caption_id': cap_id})
        annotation_json.close()
        cap_id += 1
        img_id += 1
    return data_list


def read_iiw_pairs(root_dir, split='test', finegrained=False):
    if finegrained:
        finegrained_json_path = os.path.join(root_dir, "test_annotations.json")
        with open(finegrained_json_path, 'r') as f:
            iiw_data = json.load(f)['annotations']  # should be a list of dict
        data_list = []
        for annotation in iiw_data:
            image_path = os.path.join(root_dir, annotation['image'])
            data_list.append({
                'image': image_path,
                'caption': annotation['caption'],
                'image_id': annotation['image_id'],
                'caption_id': annotation['caption_id']
            })
    else:
        data_names = ['DOCCI_Test', 'IIW-400', 'DCI_Test']
        data_subroot = {
            'DOCCI_Test': 'docci',
            'IIW-400': 'docci_aar',
            'DCI_Test': 'dci'
        }

        data_list = []
        img_id = 0
        cap_id = 0
        for data_name in data_names:
            anno_file = os.path.join(root_dir, data_name, 'data.jsonl')
            with open(anno_file, 'r') as json_file:
                anno = list(json_file)
            json_file.close()
            for data in anno:
                data = json.loads(data)
                if 'image' in data:
                    image_name = data['image']
                elif 'image/key' in data:
                    image_name = data['image/key']
                if '.jpg' not in image_name:
                    image_name += '.jpg'
                image_path = os.path.join(root_dir, data_subroot[data_name], image_name)
                caption = data['IIW']
                rejoined_caption = ". ".join(split_caption(caption))
                data_list.append({
                    'image': image_path,
                    'caption': rejoined_caption,
                    'image_id': img_id,
                    'caption_id': cap_id})
                img_id += 1
                cap_id += 1
    return data_list


def subsample(data_list, sampling_mode, num_samples):
    """
    :param data_list: List of dict [{'image_id': , 'image': , 'caption': }]
    :param sampling_mode: Str. Choice among ['random', None]
    :param num_samples: int
    :return: sampled_data_list: Subsampled list of dict
    """
    if sampling_mode == 'random':
        if num_samples > len(data_list):
            raise ValueError("num_samples cannot be greater than the length of data_list")
        sampled_data_list = random.sample(data_list, num_samples)
    else:
        sampled_data_list = data_list
    return sampled_data_list


def pre_tokenize(tokenizer, data_list):
    for data in data_list:
        data["caption"] = tokenizer(data["caption"])
    return data_list


class FlickrTextDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', tokenizer=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        #create data list
        logging.info(f"creating dataset list...")
        data_list = read_flickr_pairs(root_dir=self.root_dir, split=self.split)
        logging.info(f"dataset list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        if self.split == 'val':
            self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)
            logging.info(f"In validation mode, finish constructing the img_cap mapping dict for retrieval")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        caption = data["caption"].squeeze(dim=0)
        if self.split == 'val':
            cap_id = data["caption_id"]
            return caption, cap_id
        else:
            return caption


class FlickrImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        #create data list
        logging.info(f"reusing pre-tokenized datalist that we get from FlickrTextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


class COCOTextDataset(Dataset):
    '''
    Only loading captions and captions ID. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, dict_root_dir=None, transform=None, split='train', tokenizer=None, sampling_mode=None,
                 num_samples=None):
        self.root_dir = root_dir
        self.dict_root_dir = dict_root_dir
        self.transform = transform
        self.split = split
        self.sampling_mode = sampling_mode
        self.num_samples = num_samples
        #create data list
        logging.info(f"creating dataset list...")
        data_list = read_coco_pairs(root_dir=self.root_dir, dict_root_dir=self.dict_root_dir, split=self.split,
                                    sampling_mode=self.sampling_mode, num_samples=self.num_samples)
        logging.info(f"dataset list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        if self.split == 'val':
            self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)
            logging.info(f"In validation mode, finish constructing the img_cap mapping dict for retrieval")
            #adding two dictionaries indicating the mapping between image index and text index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        img_id = data["image_id"]

        caption = data["caption"].squeeze(dim=0)

        if self.split == 'val':
            cap_id = data["caption_id"]
            return caption, cap_id  # Only retruning captions and cap_ids
        else:
            return caption


class CC3MTextDataset(Dataset):
    '''
    Only loading captions and captions ID. Used in Text-conditioned setting. Only in validation
    Note that we are using a different retrieval mode here
    '''

    def __init__(self, root_dir, transform=None, split='train_retrieval', tokenizer=None, sampling_mode=None,
                 num_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.sampling_mode = sampling_mode
        self.num_samples = num_samples
        #create data list
        logging.info(f"creating dataset list...")
        data_list = read_cc3m_pairs(root_dir=root_dir, split=split)
        logging.info(f"dataset list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)
        #adding two dictionaries indicating the mapping between image index and text index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        caption = data["caption"].squeeze(dim=0)

        if self.split == 'train_retrieval':
            cap_id = data["caption_id"]
            return caption, cap_id  # Only retruning captions and cap_ids
        else:
            return caption


class DOCCITextDataset(Dataset):
    '''
    Only loading captions and captions ID. Used in Text-conditioned setting. Only in validation
    Note that we are using a different retrieval mode here
    '''

    def __init__(self, root_dir, transform=None, split='test', tokenizer=None, sampling_mode=None,
                 num_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.sampling_mode = sampling_mode
        self.num_samples = num_samples
        #create data list
        logging.info(f"creating DOCCI dataset list...")
        data_list = read_docci_pairs(root_dir=root_dir, split=split)
        logging.info(f"DOCCI data_list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)
        #adding two dictionaries indicating the mapping between image index and text index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        caption = data["caption"].squeeze(dim=0)

        if self.split == 'test':
            cap_id = data["caption_id"]
            return caption, cap_id  # Only retruning captions and cap_ids
        else:
            return caption


class ShareGPT4VTextDataset(Dataset):
    '''
    Only loading captions and captions ID. Used in Text-conditioned setting. Only in validation
    Note that we are using a different retrieval mode here
    '''

    def __init__(self, root_dir, total_length, transform=None, split='test', tokenizer=None, sampling_mode=None,
                 num_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.sampling_mode = sampling_mode
        self.num_samples = num_samples
        #create data list
        logging.info(f"creating ShareGPT4V dataset list...")
        share4v_json_path = os.path.join(root_dir, "share4v_sam_10k.json")
        data_list = read_sharegpt4v_pairs(root_dir=root_dir, json_name=share4v_json_path, total_len=total_length)
        logging.info(f"SHareGPT4V data_list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        caption = data["caption"].squeeze(dim=0)

        if self.split == 'test':
            cap_id = data["caption_id"]
            return caption, cap_id
        else:
            return caption


class ShareGPT4VImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        #create data list
        logging.info(f"reusing pre-tokenized datalist that we get from ShareGPT4V, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


class IIWTextDataset(Dataset):
    '''
    Only loading captions and captions ID. Used in Text-conditioned setting. Only in validation
    Note that we are using a different retrieval mode here
    '''

    def __init__(self, root_dir, transform=None, split='test', tokenizer=None, sampling_mode=None,
                 num_samples=None, finegrained=False):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.sampling_mode = sampling_mode
        self.num_samples = num_samples
        #create data list
        logging.info(f"creating IIW dataset list...")
        data_list = read_iiw_pairs(root_dir=root_dir, split=split, finegrained=finegrained)
        logging.info(f"IIW data_list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        caption = data["caption"].squeeze(dim=0)

        if self.split == 'test':
            cap_id = data["caption_id"]
            return caption, cap_id  # Only retruning captions and cap_ids
        else:
            return caption


class IIWImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        #create data list
        logging.info(f"reusing pre-tokenized datalist that we get from IIWextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


class DCITextDataset(Dataset):
    '''
    Only loading captions and captions ID. Used in Text-conditioned setting. Only in validation
    Note that we are using a different retrieval mode here
    '''

    def __init__(self, root_dir, transform=None, split='test', tokenizer=None, sampling_mode=None,
                 num_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.sampling_mode = sampling_mode
        self.num_samples = num_samples
        #create data list
        logging.info(f"creating DCI dataset list...")
        data_list = read_dci_pairs(root_dir=root_dir, split=split)
        logging.info(f"DCI data_list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        caption = data["caption"].squeeze(dim=0)

        if self.split == 'test':
            cap_id = data["caption_id"]
            return caption, cap_id  # Only retruning captions and cap_ids
        else:
            return caption


class DCIImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        #create data list
        logging.info(f"reusing pre-tokenized datalist that we get from DCITextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


class Urban1kTextDataset(Dataset):
    '''
    Only loading captions and captions ID. Used in Text-conditioned setting. Only in validation
    Note that we are using a different retrieval mode here
    '''

    def __init__(self, root_dir, transform=None, split='test', tokenizer=None, sampling_mode=None,
                 num_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.sampling_mode = sampling_mode
        self.num_samples = num_samples
        #create data list
        logging.info(f"creating urban1k dataset list...")
        data_list = read_urban1k_pairs(root_dir=root_dir, split=split)
        logging.info(f"DOCCI data_list created, pretokenizing...")
        self.data_list = pre_tokenize(tokenizer=tokenizer, data_list=data_list)
        logging.info(f"pretokenization finished...")
        self.img2txt_dict, self.txt2img_dict = map_img_cap(self.data_list)
        #adding two dictionaries indicating the mapping between image index and text index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        caption = data["caption"].squeeze(dim=0)

        if self.split == 'test':
            cap_id = data["caption_id"]
            return caption, cap_id  # Only retruning captions and cap_ids
        else:
            return caption


class Urban1kImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        #create data list
        logging.info(f"reusing pre-tokenized datalist that we get from Urban1kTextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


class CC3MImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        #create data list
        logging.info(f"reusing pre-tokenized datalist that we get from CC3MTextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


class DOCCIImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        #create data list
        logging.info(f"reusing pre-tokenized datalist that we get from DOCCITextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


def extract_unique_img_list_from_data_list(data_list):
    """
    :param data_list: a list of dicts, each: {'image', 'image_id', 'caption', 'caption_id'}
    :return: img_list: a list of dicts, with all unique 'image' w.r.t. 'image_id'. So each new dict will be {'image', 'image_id'}
    """
    seen_ids = set()
    img_list = []

    for item in data_list:
        image_id = item['image_id']
        if image_id not in seen_ids:
            # Add to the list and mark the id as seen
            img_list.append({'image': item['image'], 'image_id': image_id})
            seen_ids.add(image_id)

    return img_list


class COCOImageDataset(Dataset):
    '''
    Only loading images and img_ids. Used in Text-conditioned setting. Only in validation
    '''

    def __init__(self, root_dir, data_list=None, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        #create data list
        logging.info(f"reusing pre-tokenized datalist that we get from COCOTextDataset, extracting...")
        self.img_list = extract_unique_img_list_from_data_list(data_list=data_list)
        logging.info(f"finish extracting all unique images with img_ids from the whole data_list")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        data = self.img_list[idx]
        img_id = data["image_id"]
        img_path = data["image"]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id


def get_retrieval_coco_dataset(args, preprocess_fn, tokenizer=None, output_dict=False):
    data_root_dir = args.coco_data_root_dir
    dict_root_dir = args.dict_root_dir
    split = 'val'
    sampler = None
    shuffle = False

    txt_dataset = COCOTextDataset(root_dir=data_root_dir, dict_root_dir=dict_root_dir, transform=preprocess_fn,
                                  split=split, tokenizer=tokenizer, sampling_mode=None, num_samples=None)
    img2txt_dict, txt2img_dict = txt_dataset.img2txt_dict, txt_dataset.txt2img_dict
    num_txt_samples = len(txt_dataset)

    img_dataset = COCOImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list, transform=preprocess_fn,
                                   split=split)
    num_img_samples = len(img_dataset)

    txt_dataloader = DataLoader(
        txt_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    #TODO: This time we cannot use 'drop_last'

    txt_dataloader.num_samples = num_txt_samples
    txt_dataloader.num_batches = len(txt_dataloader)

    img_dataloader = DataLoader(
        img_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    img_dataloader.num_samples = num_img_samples
    img_dataloader.num_batches = len(img_dataloader)

    if output_dict:
        return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler), img2txt_dict, txt2img_dict
    else:
        return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler)


def get_retrieval_flickr_dataset(args, preprocess_fn, tokenizer=None, output_dict=False):
    data_root_dir = args.flickr_data_root_dir
    split = args.flickr_val_or_test
    sampler = None
    shuffle = False

    txt_dataset = FlickrTextDataset(root_dir=data_root_dir, transform=preprocess_fn,
                                    split=split, tokenizer=tokenizer)
    img2txt_dict, txt2img_dict = txt_dataset.img2txt_dict, txt_dataset.txt2img_dict
    num_txt_samples = len(txt_dataset)

    img_dataset = FlickrImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list, transform=preprocess_fn,
                                     split=split)
    num_img_samples = len(img_dataset)


    txt_dataloader = DataLoader(
        txt_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    # TODO: This time we cannot use 'drop_last'

    txt_dataloader.num_samples = num_txt_samples
    txt_dataloader.num_batches = len(txt_dataloader)

    img_dataloader = DataLoader(
        img_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    img_dataloader.num_samples = num_img_samples
    img_dataloader.num_batches = len(img_dataloader)

    if output_dict:
        return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler), img2txt_dict, txt2img_dict
    else:
        return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler)


def get_finegrained_or_long_retrieval_dataset(args, preprocess_fn, tokenizer=None, output_dict=False,
                                      dataset_name='cc3m_train'):
    sampler = None
    shuffle = False
    if dataset_name == "cc3m_train":
        split = 'train_retrieval'
        data_root_dir = args.cc3m_train_retrieval_dir
        txt_dataset = CC3MTextDataset(root_dir=data_root_dir, transform=preprocess_fn, split=split, tokenizer=tokenizer,
                                      sampling_mode=None, num_samples=None)
    elif dataset_name == "docci":
        split = 'test'
        data_root_dir = args.docci_retrieval_dir
        txt_dataset = DOCCITextDataset(root_dir=data_root_dir, transform=preprocess_fn, split=split,
                                       tokenizer=tokenizer,
                                       sampling_mode=None, num_samples=None)
    elif dataset_name == "urban-1k":
        split = 'test'
        data_root_dir = args.urban_1k_retrieval_dir
        txt_dataset = Urban1kTextDataset(root_dir=data_root_dir, transform=preprocess_fn, split=split,
                                         tokenizer=tokenizer,
                                         sampling_mode=None, num_samples=None)
    elif dataset_name == "iiw":
        split = 'test'
        data_root_dir = args.iiw_retrieval_dir
        txt_dataset = IIWTextDataset(root_dir=data_root_dir, transform=preprocess_fn, split=split, tokenizer=tokenizer,
                                     sampling_mode=None, num_samples=None, finegrained=args.use_finegrained_iiw)

    elif dataset_name == 'dci':
        split = 'test'
        data_root_dir = args.dci_retrieval_dir
        txt_dataset = DCITextDataset(root_dir=data_root_dir, transform=preprocess_fn, split=split, tokenizer=tokenizer,
                                     sampling_mode=None, num_samples=None)

    elif dataset_name == 'sharegpt4v-1k':
        split = 'test'
        data_root_dir = args.sharegpt4v_retrieval_dir
        txt_dataset = ShareGPT4VTextDataset(root_dir=data_root_dir, total_length=1000, transform=preprocess_fn,
                                            split=split, tokenizer=tokenizer,
                                            sampling_mode=None, num_samples=None)

    elif dataset_name == 'sharegpt4v-10k':
        split = 'test'
        data_root_dir = args.sharegpt4v_retrieval_dir
        txt_dataset = ShareGPT4VTextDataset(root_dir=data_root_dir, total_length=10000, transform=preprocess_fn,
                                            split=split, tokenizer=tokenizer,
                                            sampling_mode=None, num_samples=None)

    else:
        raise NotImplementedError

    img2txt_dict, txt2img_dict = txt_dataset.img2txt_dict, txt_dataset.txt2img_dict
    num_txt_samples = len(txt_dataset)

    if dataset_name == "cc3m_train":
        img_dataset = CC3MImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list, transform=preprocess_fn)
    elif dataset_name == "docci":
        img_dataset = DOCCIImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list,
                                        transform=preprocess_fn)
    elif dataset_name == "urban-1k":
        img_dataset = Urban1kImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list,
                                          transform=preprocess_fn)
    elif dataset_name == "iiw":
        img_dataset = IIWImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list,
                                      transform=preprocess_fn)

    elif dataset_name == "dci":
        img_dataset = DCIImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list,
                                      transform=preprocess_fn)

    elif dataset_name in ["sharegpt4v-1k", "sharegpt4v-10k"]:
        img_dataset = ShareGPT4VImageDataset(root_dir=data_root_dir, data_list=txt_dataset.data_list,
                                             transform=preprocess_fn)
    else:
        raise NotImplementedError

    num_img_samples = len(img_dataset)

    # drop_last = is_train or args.text_conditioned_loss
    # if we used text_conditioned_loss, then we always drop_last

    txt_dataloader = DataLoader(
        txt_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )

    txt_dataloader.num_samples = num_txt_samples
    txt_dataloader.num_batches = len(txt_dataloader)

    img_dataloader = DataLoader(
        img_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    img_dataloader.num_samples = num_img_samples
    img_dataloader.num_batches = len(img_dataloader)

    if output_dict:
        return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler), img2txt_dict, txt2img_dict
    else:
        return DataInfo(txt_dataloader, sampler), DataInfo(img_dataloader, sampler)
