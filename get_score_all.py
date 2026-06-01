import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import gc

def get_classes_names(path="./datasets/imagenet-r"):
    train_path = os.path.join(path, 'train')
    words_path = os.path.join(path, 'README.txt')

    wnids = sorted(os.listdir(train_path))
    wnid_to_words = {}
    with open(words_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                continue

            wnid, words = parts
            wnid_to_words[wnid] = words.split(',')[0].replace("_", " ")

    return [wnid_to_words[wnid] for wnid in wnids if wnid in wnid_to_words]

#CLASS_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
CLASS_ORDER = [168, 136, 51, 9, 183, 101, 171, 99, 42, 159, 191, 70, 16, 188, 27, 10, 175, 26, 68, 187, 98, 6, 85, 35, 112, 43, 100, 0, 103, 181, 88, 59, 4, 2, 116, 174, 94, 80, 106, 1, 147, 17, 141, 131, 72, 23, 173, 54, 197, 118, 87, 32, 79, 104, 91, 19, 135, 107, 178, 36, 11, 199, 142, 8, 122, 3, 28, 57, 153, 172, 190, 56, 49, 44, 97, 62, 151, 169, 194, 55, 192, 12, 189, 78, 66, 180, 15, 137, 109, 134, 92, 119, 126, 52, 170, 40, 148, 65, 144, 64, 138, 45, 77, 89, 154, 90, 71, 193, 74, 30, 113, 143, 96, 84, 67, 50, 186, 156, 69, 21, 18, 111, 108, 58, 125, 157, 150, 110, 182, 129, 166, 83, 81, 60, 13, 165, 14, 176, 63, 117, 5, 22, 145, 121, 38, 41, 82, 127, 114, 20, 31, 53, 37, 163, 196, 130, 152, 162, 86, 76, 24, 34, 184, 149, 33, 128, 198, 155, 146, 167, 139, 120, 140, 102, 47, 25, 158, 123, 46, 164, 61, 7, 115, 75, 133, 160, 105, 132, 179, 124, 48, 73, 93, 39, 95, 195, 29, 177, 185, 161]
#CLASS_ORDER = [81, 14, 3, 94, 92, 36, 49, 77, 20, 32, 21, 47, 56, 16, 71, 27, 70, 91, 57, 26, 74, 90, 79, 97, 23, 73, 87, 66, 84, 30, 24, 88, 19, 93, 29, 65, 35, 4, 86, 58, 68, 6, 0, 40, 13, 25, 75, 64, 46, 80, 28, 83, 7, 9, 12, 37, 17, 60, 31, 33, 8, 2, 63, 10, 18, 55, 95, 78, 85, 62, 96, 44, 99, 76, 11, 54, 45, 5, 89, 98, 61, 67, 69, 82, 41, 39, 42, 53, 51, 52, 50, 1, 34, 22, 38, 15, 43, 59, 72, 48]
ALL_NAMES = get_classes_names(path="./datasets/imagenet-r")


ORDERED_NAMES = [ALL_NAMES[i] for i in CLASS_ORDER]
NAME2IDX = {n: i for i, n in enumerate(ORDERED_NAMES)}

initial_increment = 20
increment = 20
task_num = 10

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

path_results = './results/'
all_name = [f for f in os.listdir(path_results) if f.startswith("test_") and f.endswith(".txt")]
all_name = sorted(all_name, key=lambda x: int(x.split('.')[0].split('_')[1]))
all_mean = []

for task_id, name in enumerate(all_name):
    with open(os.path.join(path_results, name)) as f:
        label_list = []
        msg_list = []
        lines = f.readlines()

        inner_task = [[] for _ in range(task_num)]
        idx = 0
        for line in lines:
            if line.startswith('the label is'):
                lab = line[13:].strip('\n')
                label_list.append(lab)

                for t in range(task_id + 1):
                    lo = 0 if t == 0 else initial_increment + (t - 1) * increment
                    hi = initial_increment + t * increment
                    visible_names = ORDERED_NAMES[lo:hi]
                    if lab in visible_names:
                        inner_task[t].append(idx)
                        break
                idx += 1

            if line.startswith('msg:'):
                s = line[26:].strip('\n').strip('.').strip('#')
                msg_list.append(s)

        text_tokens = clip.tokenize(ORDERED_NAMES).to(device)

        new_msg_list = msg_list

        batch_size = 64
        predict_features = []
        with torch.no_grad():
            for i in tqdm(range(0, len(new_msg_list), batch_size)):
                batch_msgs = new_msg_list[i:i + batch_size]
                batch_tokens = clip.tokenize(batch_msgs).to(device)
                batch_feat = model.encode_text(batch_tokens)
                batch_feat = batch_feat / batch_feat.norm(dim=1, keepdim=True)
                predict_features.append(batch_feat.cpu())

            predict_feature = torch.cat(predict_features).to(device)

            text_features_label = model.encode_text(text_tokens)
            text_features_label = text_features_label / text_features_label.norm(dim=1, keepdim=True)

        sim = predict_feature @ text_features_label.T

        real_label_full = []
        valid_idx = []
        for i, lab in enumerate(label_list):
            if lab in NAME2IDX:
                real_label_full.append(NAME2IDX[lab])
                valid_idx.append(i)
        if len(valid_idx) == 0:
            print(f"[WARN] {name}: no valid samples matched ORDERED_NAMES_200.")
            continue

        real_label = torch.tensor(real_label_full, device='cpu', dtype=torch.long)
        sim_valid = sim[valid_idx]
        pre = sim_valid.cpu().argmax(dim=1)
        acc = float((pre == real_label).float().mean().item())
        task_acc = []

        for t in range(task_id + 1):
            mask = [j for j in inner_task[t] if j in valid_idx]
            if len(mask) > 0:
                rel = [valid_idx.index(j) for j in mask]
                acc_i = float((pre[rel] == real_label[rel]).float().mean().item())
            else:
                acc_i = 0.0
            task_acc.append(acc_i)

        for acc_each in task_acc:
            print(str(round(acc_each * 100, 2)), end=' ')
        print('mean: ', str(round(acc * 100, 2)))

        all_mean.append(round(acc * 100, 2))

        torch.cuda.empty_cache()
        gc.collect()

print('avg: ', round(sum(all_mean) / len(all_mean), 2))
