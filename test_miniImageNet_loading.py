random.seed(10) 

# Recall we have the training and valid splits, now we do the valid and test split.

shuffled = random.sample(files_list_miniImageNet,len(files_list_miniImageNet))
trainlist_final,_ = get_training_and_valid_sets(shuffled)
_,vallist = get_training_and_valid_sets(shuffled)

# For validation and test data splitting.

def get_valid_and_test_sets(file_list):
    split = 0.50           # 20 class set as test.
    split_index = floor(len(file_list) * split)
    # valid.
    training = file_list[:split_index]
    # test.
    validation = file_list[split_index:]
    return training, validation

validlist_final,_ = get_valid_and_test_sets(vallist)
_,testlist_final = get_valid_and_test_sets(vallist)

test_img = []

for test in testlist_final:
   data_test_img = load_images(path + '/' + test + '/')
   test_img.append(data_test_img)


############# valid, test images + labels in array list format ##################

test_img_final = []
test_label_final = []

for e in range (len(test_img)):
   for f in range (600):   # Each class has 600 images.
      test_img_final.append(test_img[e][f])
      test_label_final.append(e+80)


############# Reassemble in tuple format. ##################

test_array = []

for e,f in zip(test_img_final,test_label_final):
  test_array.append((e,f))

################## shuffle #############################

test_array = shuffle(test_array)

new_X_test_miniImageNet = [x[0] for x in test_array]
new_y_test_miniImageNet = [x[1] for x in test_array]

test_dataset =  miniImageNet_CustomDataset(new_X_test_miniImageNet,new_y_test_miniImageNet, transform=[data_transform_valtest])
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

