import os
import json
import numpy as np
import argh
import boto3
from argh import arg
from scipy.spatial import distance
from plotting_funcs import *

s3 = boto3.client('s3')

def compute_dist(img_embeds, dist_func=distance.euclidean, obj='Pedestrian:1'):
    dists = []
    for i in img_embeds:
        if (i>0)&(obj in list(img_embeds[i].keys())):
            if (obj in list(img_embeds[i-1].keys())):
                dist = dist_func(img_embeds[i-1][obj],img_embeds[i][obj]) # distance  between frame at t0 and t1
        #         distance.cosine()
                dists.append(dist)
    return dists

def get_problem_frames(lab_frame, size_thresh=.25, iou_thresh=.4, embed=False, imgs=None):
    frame_res = {}
#     frame_res[frame] = {}
    for obj in list(np.unique(lab_frame.obj)): # frame_dict[frame]
        frame_res[obj] = {}
        lframe_len = max(lab_frame_real['frameid'])
        ann_subframe = lab_frame[lab_frame.obj==obj]
        fframe = min(ann_subframe['frameid'])
        lframe = max(ann_subframe['frameid'])
        size_vec = np.zeros(lframe_len+1)
        size_vec[fframe:lframe+1] = ann_subframe['height']*ann_subframe['width']
        size_diff = np.array(size_vec[:-1])- np.array(size_vec[1:])
        norm_size_diff = size_diff/np.array(size_vec[:-1])
        norm_size_diff[np.where(np.isnan(norm_size_diff))[0]] = 0
        norm_size_diff[np.where(np.isinf(norm_size_diff))[0]] = 0
        frame_res[obj]['size_diff'] = [int(x) for x in size_diff]
        frame_res[obj]['norm_size_diff'] = [int(x) for x in norm_size_diff]
        try:
            problem_frames = [int(x) for x in np.where(np.abs(norm_size_diff)>size_thresh)[0]]
#             worst_frame = np.argmax(np.abs(norm_size_diff))
#             print('Worst frame for',obj,'in',frame, 'is: ',worst_frame)
        except:
            problem_frames = []
        frame_res[obj]['size_problem_frames'] = problem_frames
        
        ious = []
        for i in range(len(lab_frame[lab_frame.obj==obj])-1):
            iou = calc_frame_int_over_union(lab_frame, obj, i)
            ious.append(iou)
        frame_res[obj]['iou'] = ious
        inds = [int(x) for x in np.where(np.array(ious)<iou_thresh)[0]]
        frame_res[obj]['iou_problem_frames'] = inds
        
        if embed:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
            model.eval()
            modules=list(model.children())[:-1]
            model=nn.Sequential(*modules)
                
            img_crops = {}
            img_embeds = {}

            for j,img in tqdm(enumerate(imgs)):
                img_arr = np.array(img)
                img_embeds[j] = {}
                img_crops[j] = {}
                # need to change this to use dataframe 
                for i,annot in enumerate(tlabels['tracking-annotations'][j]['annotations']):
                    try:
                        crop = img_arr[annot['top']:(annot['top']+annot['height']),annot['left']:(annot['left']+annot['width']),:]                    
                        new_crop = np.array(Image.fromarray(crop).resize((224,224)))
                        img_crops[j][annot['object-name']] = new_crop
                        new_crop = np.reshape(new_crop, (1,224,224,3))
                        new_crop = np.reshape(new_crop, (1,3,224,224))
                        torch_arr = torch.tensor(new_crop, dtype=torch.float)
                        with torch.no_grad():
                            emb = model(torch_arr)
                        img_embeds[j][annot['object-name']] = emb.squeeze()
                    except:
                        pass
                    
            dists = compute_dist(img_embeds, obj=obj)

            # look for distances that are 2+ standard deviations greater than the mean distance
            prob_frames = np.where(dists>(np.mean(dists)+np.std(dists)*2))[0]
            frame_res[obj]['embed_prob_frames'] = prob_frames
        
    return frame_res
    
        

# for frame in tqdm(frame_dict):
@arg('--bucket', help='s3 bucket to retrieve labels from and save result to', default=None)
@arg('--lab_path', help='s3 key for labels to be analyzed, an example would look like mot_track_job_results/annotations/consolidated-annotation/output/0/SeqLabel.json', default=None)
@arg('--size_thresh', help='Threshold for identifying allowable percentage size change for a given object between frames', default=.25)
@arg('--iou_thresh', help='Threshold for identifying the bounding boxes of objects that fall below this IoU metric between frames', default=.4)
@arg('--embed', help='Perform sequential object bounding box crop embedding comparison. Generates embeddings for the crop of a given object throughout the video and compares them sequentially, requires downloading a model from PyTorch Torchhub', default=False)
@arg('--imgs', help='Path to images to be used for sequential embedding analysis, only required if embed=True', default=None)
@arg('--save_path', help='s3 key to save quality analysis results to', default=None)
def run_quality_check(bucket = None, lab_path = None,
                       size_thresh=.25, iou_thresh=.4, embed=False, imgs=None, save_path=None):
    """
    Main data quality check utility.
    Designed for use on a single video basis, please provide a SeqLabel.json file to analyze, this can typically be found in 
    the s3 output folder for a given Ground Truth Video job under annotations > consolidated-annotation > output
    """
    
    print('downloading labels')

    s3.download_file(Bucket=bucket, Key=lab_path, Filename = 'SeqLabel.json')   
#     os.system(f'aws s3 cp s3://{bucket}/{lab_path} SeqLabel.json')
    
    with open('SeqLabel.json', 'r') as f:
        tlabels = json.load(f)
    lab_frame_real = create_annot_frame(tlabels['tracking-annotations'])
    
    print('Running analysis...')
    frame_res = get_problem_frames(lab_frame_real, size_thresh=size_thresh, iou_thresh=iou_thresh, embed=embed)
    
    with open('quality_results.json', 'w') as f:
        json.dump(frame_res, f)
    
    print(f'Output saved to s3 path s3://{bucket}/{save_path}')
    s3.upload_file(Bucket=bucket, Key=save_path, Filename='quality_results.json')

#     os.system(f'aws s3 cp quality_results.json s3://{bucket}/{save_path}')
    
        
def main():
    parser = argh.ArghParser()
    parser.add_commands([run_quality_check])
    parser.dispatch()
    
if __name__ == "__main__":
    main()

    

# # can try with resizing to 224x224, or filling in with 0s

# img_crops = {}
# img_embeds = {}

# for j,img in tqdm(enumerate(imgs[:32])):
#     img_arr = np.array(img)
#     img_embeds[j] = {}
#     img_crops[j] = {}
#     for i,annot in enumerate(tlabels['tracking-annotations'][j]['annotations']):
#         crop = img_arr[annot['top']:(annot['top']+annot['height']),annot['left']:(annot['left']+annot['width']),:]
#         try:
#             new_crop = np.array(Image.fromarray(crop).resize((224,224)))
#             img_crops[j][annot['object-name']] = new_crop
#             new_crop = np.reshape(new_crop, (1,224,224,3))
#             new_crop = np.reshape(new_crop, (1,3,224,224))
#             torch_arr = torch.tensor(new_crop, dtype=torch.float)
#             embed = resnext50_32x4d(torch_arr)
#             img_embeds[j][annot['object-name']] = embed.squeeze()
#         except:
#             print(crop.shape)


# can try with resizing to 224x224, or filling in with 0s

# img_crops = {}
# img_embeds = {}

# for j,img in tqdm(enumerate(imgs[:32])):
#     img_arr = np.array(img)
#     img_embeds[j] = {}
#     img_crops[j] = {}
#     for i,annot in enumerate(tlabels['tracking-annotations'][j]['annotations']):
#         crop = img_arr[annot['top']:(annot['top']+annot['height']),annot['left']:(annot['left']+annot['width']),:]
#         new_crop = np.zeros((224,224,3))
#         new_crop[:crop.shape[0],:crop.shape[1],:] = crop[:224,:224,:]
#         img_crops[j][annot['object-name']] = new_crop
#         new_crop = np.reshape(new_crop, (1,224,224,3))
#         new_crop = np.reshape(new_crop, (1,3,224,224))
#         torch_arr = torch.tensor(new_crop, dtype=torch.float)
#         embed = resnext50_32x4d(torch_arr)
#         img_embeds[j][annot['object-name']] = embed.squeeze()

        
# new_crop = np.zeros((224,224,3))
# new_crop[:crop.shape[0],:crop.shape[1],:] = crop
# imshow(new_crop)


# dists = []
# for i in img_embeds:
#     if i>0:
#         dist = distance.euclidean(img_embeds[i]['Pedestrian:1'],img_embeds[i-1]['Pedestrian:1'])
#         dists.append(dist)
        
        
# avg_embed = []
# for i in img_embeds:
#     avg_embed.append(img_embeds[i]['Pedestrian:1'].detach().cpu().numpy())
    
# avg_embed = np.mean(avg_embed, axis=0)
    
# avg_dists = []
# for i in img_embeds:
#     dist = distance.euclidean(img_embeds[i]['Pedestrian:1'],avg_embed)
#     avg_dists.append(dist)