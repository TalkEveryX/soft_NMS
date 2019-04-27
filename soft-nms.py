#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import xml.etree.cElementTree as ET
import random

def nms(boxes, scores, iou_threshold, max_output_size,soft_nms=False):
    
    keep = []
    order = scores.argsort()[::-1]#按得分从大到小排序
    num = boxes.shape[0]
    suppressed = np.zeros((num), dtype=np.int)#抑制
    for _i in range(num):
        if len(keep) >= max_output_size:
            break
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
	####boxes左上和右下角坐标####
        xi1=boxes[i, 0]
        yi1=boxes[i, 1]
        xi2=boxes[i, 2]
        yi2=boxes[i, 3]
        areas1=(xi2 - xi1 + 1) * (yi2 - yi1 + 1)#box1面积
        for _j in range(_i + 1, num):#start，stop
            j = order[_j]
            if suppressed[i] == 1:
                continue
            xj1=boxes[j, 0]
            yj1=boxes[j, 1]
            xj2=boxes[j, 2]
            yj2=boxes[j, 3]
            areas2=(xj2 - xj1 + 1) * (yj2 - yj1 + 1)#box2面积
            
            xx1 = np.maximum(xi1, xj1) 
            yy1 = np.maximum(yi1, yj1)
            xx2 = np.minimum(xi2, xj2)
            yy2 = np.minimum(yi2, yj2)
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            
            int_area=w*h#重叠区域面积
            
            inter = 0.0
          
            if int_area>0:
                inter = int_area * 1.0 / (areas1 + areas2 - int_area)#IOU
            ###softnms
	    if soft_nms:
                sigma=0.6
                if inter >= iou_threshold:
                    scores[j]=np.exp(-(inter * inter)/sigma)*scores[j]
            ###nms
            else:    
                if inter >= iou_threshold:
                    suppressed[j] = 1
    return keep#返回保留下来的下标
def read_xml_gtbox(xml_path):

    ###读取xml文件中boxes坐标值

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:  
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            for child_item in child_of_root:                
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(int(node.text))  
                    box_list.append(tmp_box)
    #box_list= np.array(box_list, dtype=np.int32)
    return img_height, img_width, box_list



if __name__ == '__main__':
    img = cv2.imread('horse.jpeg')
    img1=img.copy()
    img2=img.copy()
    h,w,gtboxes=read_xml_gtbox('horse.xml')
    ###造数据
    create_box=[]
    for i,label in enumerate(gtboxes):
        for j in range(3):
            temp=[]
            scorand=random.uniform(0.6,0.8)
            for coordinate in label:
                cord=random.randint(-30,30)
                coordinate+=cord
                temp.append(coordinate)
            temp.append(scorand)
            create_box.append(temp)
        k=random.uniform(0.95,1.0)
        label.append(k)
        create_box.append(label)
    
    print('done')
    create_box=np.array(create_box,dtype=np.float32)
    gtboxes_1=create_box[:,0:4]
    gtboxes_1.astype(int)
    scores=create_box[:,4]
    
    for lab in zip(gtboxes_1,scores):
        print(lab)
        draw=cv2.rectangle(img,(lab[0][0],lab[0][1]),(lab[0][2],lab[0][3]),(255,0,0),1)
        draw=cv2.putText(img, str(lab[1]), (lab[0][0],lab[0][1]), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0,255,255),1)
    cv2.imwrite('multiboxes.jpg', draw)
    
    keeped=nms(gtboxes_1,scores,0.4,3,False)
    gtboxes_2=gtboxes_1[keeped]
    scores_2=scores[keeped]
    for lab in zip(gtboxes_2,scores_2):
        print(lab)
        draw_1=cv2.rectangle(img1,(lab[0][0],lab[0][1]),(lab[0][2],lab[0][3]),(255,0,0),1)
        draw_1=cv2.putText(img1, str(lab[1]), (lab[0][0],lab[0][1]), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0,255,255),1)
    cv2.imwrite('result_nms.jpg', draw_1)
    
    keeped=nms(gtboxes_1,scores,0.3,3,True)
    gtboxes_2=gtboxes_1[keeped]
    scores_2=scores[keeped]
    for lab in zip(gtboxes_2,scores_2):
        print(lab)
        draw_2=cv2.rectangle(img2,(lab[0][0],lab[0][1]),(lab[0][2],lab[0][3]),(255,0,0),1)
        draw_2=cv2.putText(img2, str(lab[1]), (lab[0][0],lab[0][1]), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0,255,255),1)
    cv2.imwrite('result_softnms.jpg', draw_2)
    print('ok')
