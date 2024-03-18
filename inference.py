import json
import os
import random
import logging
import requests
import torch
from tqdm import tqdm
from modelscope import snapshot_download, AutoModel, AutoTokenizer,AutoModelForCausalLM
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)
requests.packages.urllib3.disable_warnings()

class Evaluator:
    def __init__(self, args):
        self.data_path = args.data_path
        self.img_path = args.img_path
        self.lang = args.lang
        self.model_name = args.model

        # 创建一个logger
        self.logger = logging.getLogger('MME_Evaluation_Logger')
        self.logger.setLevel(logging.DEBUG)

        # 创建一个handler，用于写入日志文件
        # 如果文件夹不存在，则创建
        self.log_path = os.path.join(args.log_path, self.model_name+"_mme.log")
        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))
        fh = logging.FileHandler(self.log_path)
        fh.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)

        self.load_model()
    
    def load_model(self):
        if self.model_name == "qwen-vl":
            self.access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjcxYjRkMmMxLWQzMDktNDgxNS05NmUxLWY0MzFmMmM0MGYzZCIsInVzZXJUeXBlIjowLCJ1c2VybmFtZSI6InNmeSIsIm5pY2tuYW1lIjoic2Z5IiwiYnVmZmVyVGltZSI6MTcwNjE3NDgxNSwiZXhwIjoxNzA4NzU5NjE1LCJqdGkiOiI5MTA2NmMyNzRkMjI0ZWY0OWE3MDZkMzVkNzkwMGIwNiIsImlzcyI6IjcxYjRkMmMxLWQzMDktNDgxNS05NmUxLWY0MzFmMmM0MGYzZCIsIm5iZiI6MTcwNjE2NzYxNSwic3ViIjoia29uZyJ9.z1GWTgsHcTE-yYLgrJ1wG3KI2L_Mzu7hYmT4Aif8A4c"
        elif self.model_name=="Xcomposer2-vl":
            torch.set_grad_enabled(False)
            # init model and tokenizer
            model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b')
            self.model = AutoModel.from_pretrained(model_dir,device_map="cuda:5", trust_remote_code=True).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            self.model.tokenizer = self.tokenizer
        elif self.model_name == "qwen-vl-int4":
            model_path = "/home/jovyan/.cache/modelscope/hub/qwen/Qwen-VL-Chat-Int4"
            # 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
        elif self.model_name == "qwen-vl-chat":
            model_path = "/home/jovyan/.cache/modelscope/hub/qwen/Qwen-VL-Chat"
            # 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
            

    def send_request_qwen_vl(self, prompt, img_path):
        url='https://122.13.25.19:5001/openapi/v1/qwenvl_ft'
        header={"Authorization":'Bearer '+self.access_token}
        data={"prompt":prompt} # 只传prompt
        if img_path is not None:
            img={"img":open(img_path,'rb')}
            r1=requests.post(url,data=data,headers=header,verify=False,files=img)
        else:
            r1=requests.post(url,data=data,headers=header,verify=False)
        r1_json = json.loads(r1.text)
        # {"code": 0, "message": "success", "result": {"text": "这是一张夜间通过隧道的监控照片，可以看到隧道内有三辆车辆通过，道路两旁有绿色的灯光照射。"}}
        # 获取text中的字符串,用get
        result = r1_json.get('result', {}).get('text', '')
        return result

    def get_access_token(self):
        token_url='https://122.13.25.19:7776/service/api/v1/oauth/91066c274d224ef49a706d35d7900b06/token'
        header={'Content-type': 'application/json; charset=UTF-8'}
        data={"grant_type":'client_credentials', 'client_id':'5735a618ac814b41af804ee7c2980d0a', 'client_secret':'9d6432c837fa44a19e3283f088dedf87'} # 只传prompt
        r1=requests.post(token_url,data=json.dumps(data),headers=header,verify=False)
        print(r1.text)
        r1_json = json.loads(r1.text)
        self.access_token = r1_json['data']['access_token']
    
    def model_inference(self, prompt, img_path):
        # --------- 生成预测结果 ---------
        self.logger.info("----------------------------------------------")
        if self.model_name == "qwen-vl":
            pred = self.send_request_qwen_vl(prompt, img_path)
        elif self.model_name == "Xcomposer2-vl":
            query = '<ImageHere>'+prompt
            with torch.cuda.amp.autocast(): 
                pred, _ = self.model.chat(self.tokenizer, query=query, image=img_path, history=[], do_sample=False)
        elif self.model_name == "qwen-vl-int4" or self.model_name == "qwen-vl-chat":
            query = self.tokenizer.from_list_format([
                {'image': img_path}, # Either a local path or an url
                {'text': prompt},
            ])
            pred, _ = self.model.chat(self.tokenizer, query=query, history=None)
        else:
            if random.random() < 0.5:
                pred = 'no, it is not'
            else:
                pred = 'yes, it is'  
        return pred

    def mme_evaluate(self):
        with open(self.data_path, 'r',encoding='utf-8') as f:
            data = json.load(f)
        accuracy_total = 0
        accuracy_plus_total = 0
        accuracy_class = {}
        accuracy_plus_class = {}

        count_total_qa = 0
        count_total_pic = 0
        count_total_class_qa = {}
        count_total_class_pic = {}
        for item in tqdm(data):
            eval_class = item.get('class',None)
            imgs = item.get('imgs',None)
            if eval_class is None or imgs is None:
                raise KeyError('class or imgs is None')
            accuracy_class[eval_class] = 0
            accuracy_plus_class[eval_class] = 0
            count_total_class_qa[eval_class] = 0
            count_total_class_pic[eval_class] = 0
            for img in imgs:
                file_name = img.get('file_name',None)
                if file_name is None:
                    raise KeyError('file_name is None')
                file_path = os.path.join(self.img_path, eval_class, file_name)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f'file_path {file_path} is not found')
                count_total_pic += 1
                correct_yes = 0
                correct_no = 0
                for qa in img.get('QA',[]):
                    language = qa.get('language', "")
                    if language != self.lang:
                        continue
                    prompt = qa.get('prompt', "")
                    label = qa.get('label', "")
                    if prompt is None or self.lang is None or label is None:
                        raise KeyError('prompt or language or label is None')
                    if self.lang == 'CN':
                        prompt += "请务必只用'是'或'否'回答此问题, 不要输出其它的。"
                    elif self.lang == 'EN':
                        prompt += "Please only answer 'yes' or 'no' to this question, do not output anything else."
                    
                    if prompt is None or self.lang is None or label is None:
                        raise KeyError('prompt or language or label is None')
                    # --------- 生成预测结果 ---------
                    pred = self.model_inference(prompt, file_path)
                    self.logger.info(f'Image: {file_path}, \nPrompt: {prompt}, \nModel output: {pred}\nLabel: {label}\n')
                    # --------------------------------
                    count_total_qa += 1
                    count_total_class_qa[eval_class] += 1
                    # 正确！
                    if pred[:len(label)].lower() == label:
                        if label == 'yes' or label == '是':
                            correct_yes +=1
                        elif label == 'no' or label == '否':
                            correct_no +=1

                count_total_class_pic[eval_class] += 1
                accuracy_class[eval_class] += correct_yes + correct_no
                accuracy_plus_class[eval_class] += 1 if correct_yes + correct_no == 2 else 0
                accuracy_total += correct_yes + correct_no
                accuracy_plus_total += 1 if correct_yes + correct_no == 2 else 0

        accuracy_total /= float(count_total_qa)
        accuracy_plus_total /= float(count_total_pic)

        for eval_class in accuracy_class.keys():
            accuracy_class[eval_class] /= float(count_total_class_qa[eval_class])
            accuracy_plus_class[eval_class] /= float(count_total_class_pic[eval_class])
        self.logger.info('Accuracy: '+str(round(accuracy_total,4)))
        self.logger.info('Accuracy+: '+str(round(accuracy_plus_total,4)))
        for eval_class in accuracy_class.keys():
            self.logger.info('Accuracy('+eval_class+'): '+str(round(accuracy_class[eval_class],4)))
            self.logger.info('Accuracy+('+eval_class+'): '+str(round(accuracy_plus_class[eval_class],4)))
        print('Accuracy:', round(accuracy_total,4))
        print('Accuracy+:', round(accuracy_plus_total,4))
        print('Accuracy(class):', accuracy_class)
        print('Accuracy+(class):', accuracy_plus_class)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/jovyan/yifeng/data/VQA_benchmarks/MME/data.json')
    parser.add_argument('--img_path', type=str, default='/home/jovyan/yifeng/data/VQA_benchmarks/MME/images')
    parser.add_argument('--lang', type=str, default='EN')
    parser.add_argument('--model', type=str, default='qwen-vl-int4',choices=['qwen-vl', 'Xcomposer2-vl', "qwen-vl-int4", "qwen-vl-chat"])
    parser.add_argument('--log_path', type=str, default='./logs')
    args = parser.parse_args()
    evaluator = Evaluator(args)
    evaluator.mme_evaluate()
    