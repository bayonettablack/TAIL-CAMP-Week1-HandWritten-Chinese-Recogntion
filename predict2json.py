import json

data_dir = '/home/kesci/work/test'
img_width, img_height = 64, 64

model = load_model("/home/kesci/work/ConvNet-small-best.h5")

test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(data_dir,
                                                  target_size = (img_width, img_height),
                                                  batch_size = 1,
                                                  color_mode = 'rgb',
                                                  shuffle = False)
num_img = len(test_generator.filenames)
pred = model.predict_generator(test_generator, steps = num_img)
predict_label = np.argmax(pred, axis = 1)
result = []
for i in range(num_img):
    filename = test_generator.filenames[i].split('/')[-1]
    label = int(predict_label[i])
    Dict = {"filename":filename,"label":label}
    result.append(Dict)
with open('ConvNet_small.json','w') as f:
    json.dump(result, f)
jsr = json.dumps(result)
print(jsr)