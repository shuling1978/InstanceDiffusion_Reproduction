import os
import argparse
from typing import List, Dict, Tuple, Optional

import PIL
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
import open_clip
from transformers import CLIPModel, CLIPTokenizer, CLIPFeatureExtractor


class CLIPEvaluator:
    """CLIP模型评估器，用于计算图像和文本的相似度"""
    
    def __init__(self, use_open_clip: bool = True):
        """
        初始化CLIP评估器
        Args:
            use_open_clip: 是否使用open_clip模型 (默认为True)
        """
        self.use_open_clip = use_open_clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_models()
        
    def _initialize_models(self):
        """初始化CLIP模型和预处理工具"""
        if not self.use_open_clip:
            # 使用HuggingFace的CLIP模型
            version = "openai/clip-vit-large-patch14"
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.model = CLIPModel.from_pretrained(version).to(self.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(version)
        else:
            # 使用OpenCLIP模型
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='laion2b-s32b-b82k')
            self.model = self.model.to(self.device)
            self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        编码文本为特征向量
        Args:
            text: 输入文本
        Returns:
            文本特征向量
        """
        if self.use_open_clip:
            text_token = self.tokenizer(text).to(self.device)
            return self.model.encode_text(text_token)
        else:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            return self.model.get_text_features(inputs["input_ids"])
    
    def encode_image(self, image: PIL.Image) -> torch.Tensor:
        """
        编码图像为特征向量
        Args:
            image: PIL图像对象
        Returns:
            图像特征向量
        """
        if self.use_open_clip:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            return self.model.encode_image(image)
        else:
            inputs = self.feature_extractor(image)
            inputs['pixel_values'] = torch.tensor(inputs['pixel_values'][0][None]).to(self.device)
            return self.model.get_image_features(inputs['pixel_values'])
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        归一化特征向量
        Args:
            features: 输入特征向量
        Returns:
            归一化后的特征向量
        """
        return features / torch.linalg.norm(features, axis=-1, keepdims=True)
    
    def compute_similarity(self, img_features: torch.Tensor, txt_features: torch.Tensor) -> float:
        """
        计算图像和文本特征的相似度
        Args:
            img_features: 图像特征
            txt_features: 文本特征
        Returns:
            相似度分数
        """
        return (img_features * txt_features).sum(axis=-1).cpu().numpy().item()


class COCODataLoader:
    """COCO数据集加载器"""
    
    def __init__(self, coco_annotations_path: str):
        """
        初始化COCO数据加载器
        Args:
            coco_annotations_path: COCO标注文件路径
        """
        self.coco = COCO(coco_annotations_path)
        self.image_ids = sorted(self.coco.getImgIds())  # 排序保证可重复性
        
    def get_image_info(self, img_id: int) -> Dict:
        """
        获取图像信息
        Args:
            img_id: 图像ID
        Returns:
            图像信息字典
        """
        return self.coco.loadImgs([img_id])[0]
    
    def get_annotations(self, img_id: int) -> List[Dict]:
        """
        获取图像标注信息
        Args:
            img_id: 图像ID
        Returns:
            标注信息列表
        """
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        return self.coco.loadAnns(ann_ids)
    
    def get_category_names(self, cat_ids: List[int]) -> List[str]:
        """
        获取类别名称
        Args:
            cat_ids: 类别ID列表
        Returns:
            类别名称列表
        """
        cats = self.coco.loadCats(cat_ids)
        return [cat["name"] for cat in cats]


class AttributeBinderEvaluator:
    """属性绑定评估器，用于评估模型对颜色/纹理等属性的绑定能力"""
    
    # 预定义颜色和纹理列表
    COLOR_LIST = ["black", "white", "red", "green", "yellow", "blue", "pink", "purple"]
    TEXTURE_LIST = ["rubber", "fluffy", "metallic", "wooden", "plastic", "fabric", "leather", "glass"]
    
    def __init__(self, args: argparse.Namespace):
        """
        初始化评估器
        Args:
            args: 命令行参数
        """
        self.args = args
        self.clip_evaluator = CLIPEvaluator()
        self._prepare_label_prompts()  # 准备标签提示
        
    def _prepare_label_prompts(self):
        """根据测试类型准备标签提示"""
        if self.args.test_random_colors:
            self.label_list = self.COLOR_LIST
            self.label_prompts = [f"a {color} object" for color in self.label_list]
        elif self.args.test_random_textures:
            self.label_list = self.TEXTURE_LIST
            self.label_prompts = [f"a {texture} object" for texture in self.texture_list]
        else:
            self.label_prompts = None
            
        if self.label_prompts:
            self.label_prompts_feats = self._encode_label_prompts(self.label_prompts)
    
    def _encode_label_prompts(self, prompts: List[str]) -> torch.Tensor:
        """
        编码标签提示为特征向量
        Args:
            prompts: 提示文本列表
        Returns:
            堆叠后的特征张量
        """
        encoded_prompts = []
        for prompt in prompts:
            text_token = self.clip_evaluator.tokenizer(prompt).to(self.clip_evaluator.device)
            features = self.clip_evaluator.model.encode_text(text_token)
            normalized = self.clip_evaluator.normalize_features(features)
            encoded_prompts.append(normalized)
        return torch.stack(encoded_prompts)
    
    def evaluate_image(self, image_path: str, phrases: List[str], locations: List[List[float]]) -> Tuple[float, float]:
        """
        评估单张图像
        Args:
            image_path: 图像路径
            phrases: 文本描述列表
            locations: 边界框位置列表
        Returns:
            (平均CLIP分数, 平均准确率)
        """
        try:
            image = Image.open(image_path).convert("RGB")
            cropped_images = self._crop_images(image, locations)
            
            clip_scores = []
            clip_accs = []
            
            for text, cropped_img in zip(phrases, cropped_images):
                score, acc = self._evaluate_instance(text, cropped_img)
                clip_scores.append(score)
                clip_accs.append(acc)
                
            return np.mean(clip_scores) if clip_scores else 0, np.mean(clip_accs) if clip_accs else 0
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            return 0, 0
    
    def _crop_images(self, image: PIL.Image, locations: List[List[float]]) -> List[PIL.Image]:
        """
        根据边界框裁剪图像
        Args:
            image: 原始图像
            locations: 边界框位置列表(归一化坐标)
        Returns:
            裁剪后的图像列表
        """
        cropped_images = []
        for location in locations:
            x0, y0, x1, y1 = location
            cropped = image.crop((
                x0 * image.width, 
                y0 * image.height, 
                x1 * image.width, 
                y1 * image.height
            ))
            cropped_images.append(cropped)
        return cropped_images
    
    def _evaluate_instance(self, text: str, image: PIL.Image) -> Tuple[float, float]:
        """
        评估单个实例(图像+文本)
        Args:
            text: 文本描述
            image: 图像对象
        Returns:
            (CLIP分数, 准确率)
        """
        with torch.no_grad():
            # 编码文本和图像特征
            txt_features = self.clip_evaluator.encode_text(text)
            img_features = self.clip_evaluator.encode_image(image)
            
            # 归一化特征
            img_features, txt_features = [
                self.clip_evaluator.normalize_features(x) 
                for x in [img_features, txt_features]
            ]
            
            # 计算CLIP分数
            clip_score = self.clip_evaluator.compute_similarity(img_features, txt_features)
            
            # 如果是属性测试，计算准确率
            clip_acc = 0
            if self.label_prompts_feats is not None:
                gt_attr = text.split(" ")[0]  # 获取属性前缀
                gt_idx = self.label_list.index(gt_attr)
                similarity = (img_features * self.label_prompts_feats).sum(axis=-1).cpu().numpy()
                pred = np.argmax(similarity)
                clip_acc = 1 if pred == gt_idx else 0
                
        return clip_score, clip_acc


def convert_coco_box(bbox: List[float], img_info: Dict) -> List[float]:
    """
    将COCO边界框格式转换为PIL.Image格式(归一化坐标)
    Args:
        bbox: COCO格式边界框 [x,y,width,height]
        img_info: 图像信息字典
    Returns:
        归一化边界框 [x0,y0,x1,y1]
    """
    x0 = bbox[0] / img_info['width']
    y0 = bbox[1] / img_info['height']
    x1 = (bbox[0] + bbox[2]) / img_info['width']
    y1 = (bbox[1] + bbox[3]) / img_info['height']
    return [x0, y0, x1, y1]


def process_annotations(annotations: List[Dict], img_info: Dict, args: argparse.Namespace, max_objs: int = 30) -> Dict:
    """
    处理标注信息
    Args:
        annotations: 原始标注列表
        img_info: 图像信息
        args: 命令行参数
        max_objs: 最大处理对象数
    Returns:
        处理后的信息字典
    """
    test_info = {
        'phrases': None,
        'locations': None,
        'file_name': img_info['file_name']
    }
    
    # 获取边界框并转换格式
    bbox_list = [ann["bbox"] for ann in annotations]
    test_info['locations'] = [convert_coco_box(bbox, img_info) for bbox in bbox_list][:max_objs]
    
    # 获取类别ID
    cat_ids = [ann['category_id'] for ann in annotations]
    
    # 根据测试类型添加属性前缀
    cat_names = COCODataLoader.get_category_names(cat_ids)
    if args.test_random_colors:
        cat_inst_ids = [ann['id'] for ann in annotations]
        colors = [AttributeBinderEvaluator.COLOR_LIST[cat_inst_id % len(AttributeBinderEvaluator.COLOR_LIST)] 
                 for cat_inst_id in cat_inst_ids]
        cat_names = [f"{color} {name}" for name, color in zip(cat_names, colors)]
    elif args.test_random_textures:
        cat_inst_ids = [ann['id'] for ann in annotations]
        textures = [AttributeBinderEvaluator.TEXTURE_LIST[cat_inst_id % len(AttributeBinderEvaluator.TEXTURE_LIST)] 
                   for cat_inst_id in cat_inst_ids]
        cat_names = [f"{texture} {name}" for name, texture in zip(cat_names, textures)]
    
    test_info['phrases'] = cat_names[:max_objs]
    return test_info


def main(args):
    """主函数"""
    # 初始化数据加载器
    coco_loader = COCODataLoader('datasets/coco/annotations/instances_val2017.json')
    
    # 初始化评估器
    evaluator = AttributeBinderEvaluator(args)
    
    # 准备结果存储
    clip_score_list = []
    clip_acc_list = []
    
    # 处理每张图像
    for img_id in tqdm(coco_loader.image_ids, desc="Processing images"):
        img_info = coco_loader.get_image_info(img_id)
        annotations = coco_loader.get_annotations(img_id)
        
        # 处理标注信息
        test_info = process_annotations(annotations, img_info, args)
        
        # 构建图像路径
        image_path = os.path.join("generation_samples/", args.folder, img_info['file_name'])
        
        # 评估图像
        avg_score, avg_acc = evaluator.evaluate_image(
            image_path, 
            test_info['phrases'], 
            test_info['locations']
        )
        
        # 存储结果
        if avg_score > 0:
            clip_score_list.append(avg_score)
        if avg_acc > 0:
            clip_acc_list.append(avg_acc)
    
    # 输出结果
    if clip_acc_list:
        print(f"平均准确率: {np.mean(clip_acc_list):.4f}")
    if clip_score_list:
        print(f"平均CLIP分数: {np.mean(clip_score_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int, default=0, help="任务索引")
    parser.add_argument("--num_jobs", type=int, default=1, help="总任务数")
    parser.add_argument("--folder", type=str, default="cocoval17-0202-308082-75%grounding-captions-InstSampler-Step15-Mean-colors", 
                       help="生成图像文件夹")
    parser.add_argument("--test_random_colors", action='store_true', 
                       help="测试随机颜色绑定的准确性")
    parser.add_argument("--test_random_textures", action='store_true', 
                       help="测试随机纹理绑定的准确性")
    
    args = parser.parse_args()
    main(args)