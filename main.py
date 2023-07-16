from utils import preprocessing_dataset
import training.work_with_data as work_with_data
import training.train_Xception as train_Xception
import logging
import matplotlib.pyplot as plt
import time

model_import = False
train = True
finetuning = False
data_prepare = False

logging.basicConfig(filename="./output/firenet.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

# create an object from preprocessing_dataset class
# create the preprocessed dataset, resize the images and save them in the destination directory
if data_prepare == True:
    logging.info("PREPROCESSING DATASET")
    print("PREPROCESSING DATASET")
    start = time.time()
    preprocessing_dataset = preprocessing_dataset.prepr_dat()
    preprocessing_dataset.split_dataset()
    end = time.time()
    logging.info("DATASET PREPROCESSED IN {} SECONDS".format(end - start))
    print("DATASET PREPROCESSED IN {} SECONDS".format(end - start))

# create an object from load_data class
logging.info("LOADING DATASET")
print("LOADING DATASET")
start = time.time()
load_data_obj = work_with_data.load_data()
# load the dataset
train_ds = load_data_obj.create_train_dataloader()
val_ds = load_data_obj.create_validation_dataloader()
test_ds = load_data_obj.create_test_dataloader()
end = time.time()
logging.info("DATASET LOADED IN {} SECONDS".format(end - start))
print("DATASET LOADED IN {} SECONDS".format(end - start))

# class names
class_names = train_ds.class_names
logging.info("CLASS NAMES: {}".format(class_names))
print("CLASS NAMES: {}".format(class_names))

# check the shape of the data
for image_batch, labels_batch in train_ds:
    logging.info("IMAGE BATCH SHAPE: {}".format(image_batch.shape))
    print("IMAGE BATCH SHAPE: {}".format(image_batch.shape))
    break

# visualize some data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# save the figure in output directory
plt.savefig("./output/visualize_data.png")
logging.info("VISUALIZE DATA SAVED IN ./output/visualize_data.png")
print("VISUALIZE DATA SAVED IN ./output/visualize_data.png")

# data augmentation
logging.info("DATA AUGMENTATION")
print("DATA AUGMENTATION")
start = time.time()
train_ds = load_data_obj.prepare(train_ds, shuffle=True, augment=True)
val_ds = load_data_obj.prepare(val_ds, shuffle=True)
end = time.time()
logging.info("DATA AUGMENTATION DONE IN {} SECONDS".format(end - start))
print("DATA AUGMENTATION DONE IN {} SECONDS".format(end - start))

# create an object from train_model class
logging.info("CREATING MODEL")
print("CREATING MODEL")
start = time.time()
train_model_obj = train_Xception.Xception()
if model_import == False:
    model = train_model_obj.build_model()
    end = time.time()
    logging.info("MODEL CREATED IN {} SECONDS".format(end - start))
    print("MODEL CREATED IN {} SECONDS".format(end - start))
    print(model.summary())
else:
    # load the model
    model = train_model_obj.load_model("./output/model-trained.keras")
    end = time.time()
    logging.info("MODEL LOADED IN {} SECONDS".format(end - start))
    print("MODEL LOADED IN {} SECONDS".format(end - start))
    print(model.summary())

# save model summary in a txt file in output directory
with open("./output/model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
logging.info("MODEL SUMMARY SAVED IN ./output/model_summary.txt")

if train == True:
    # train the model with backbone freezed
    logging.info("TRAINING MODEL WITH BACKBONE FREEZED")
    print("TRAINING MODEL WITH BACKBONE FREEZED")
    start = time.time()
    hystory, model = train_model_obj.train(
        model, train_ds, val_ds, NUM_EPOCHS=10)
    end = time.time()
    logging.info("MODEL TRAINED IN {} SECONDS".format(end - start))
    logging.info("LOG OF TRAINING SAVED IN ./output/training_log.csv")
    print("MODEL TRAINED IN {} SECONDS".format(end - start))
    train_model_obj.save_model(model, "./output/model-trained.keras")
    load_data_obj.create_training_chart(
        19, hystory, "./output/training_plot.png")

if finetuning == True:
    # now finetune the model unfreezing the backbone
    logging.info("FINETUNING MODEL UNFREEZING THE BACKBONE")
    print("FINETUNING MODEL UNFREEZING THE BACKBONE")
    start = time.time()
    hystory1, model = train_model_obj.finetuning(
        model, train_ds, val_ds, NUM_EPOCHS=10)
    end = time.time()
    logging.info("MODEL FINETUNED IN {} SECONDS".format(end - start))
    logging.info(
        "LOG OF FINETUNING SAVED IN ./output/training_finetuning_log.csv")
    print("MODEL FINETUNED IN {} SECONDS".format(end - start))
    train_model_obj.save_model(model, "./output/model-finetuned.h5")
    load_data_obj.create_training_chart(
        10, hystory1, "./output/training_finetuning_plot.png")

# evaluate the model
logging.info("EVALUATING MODEL")
print("EVALUATING MODEL")
start = time.time()
loss, acc, recall = train_model_obj.evaluate(model, test_ds)
end = time.time()
logging.info("MODEL EVALUATED IN {} SECONDS WITH LOSS {} AND ACCURACY {} AND RECALL {}".format(
    end - start, loss, acc, recall))
print("MODEL EVALUATED IN {} SECONDS WITH LOSS {} AND ACCURACY {} AND RECALL {}".format(
    end - start, loss, acc, recall))
