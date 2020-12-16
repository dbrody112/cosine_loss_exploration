from chexpert import showImage

class FlowersDataset(Dataset):
    def __init__(self,csv_file, train_csv, file_paths, transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomAffine(degrees = 30,translate = (0.1,0.1)),
    transforms.ToTensor(),
    ])):
    
        self.csv_file = csv_file
        self.train_csv = train_csv
        self.file_paths = file_paths
        self.transform = transform
    
    def __len__(self):
        return(len(self.csv_file))
    def __getitem__(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.file_paths[idx])
        img = np.array(img)
        img = transforms.functional.to_pil_image(img)
        if self.transform:
            img = self.transform(img)
        
        unique_classes = [i+1 for i in range(len(self.train_csv['labels'].unique()))]
        class_dict = {}
        for i in range(len(self.train_csv['labels'].unique())):
            z = torch.zeros(1,max(unique_classes))
            z[:,i] = 1
            class_dict[self.train_csv['labels'].unique()[i]] = z
        label = torch.FloatTensor(class_dict[self.csv_file['labels'][idx]])
        img.to(device)
        label.to(device)
        sample = [torch.FloatTensor(img).to(device),label.to(device)]

        return sample

def loadFlowerDataset(image_labels_path = './imagelabels.mat', train_test_validation_split_path = './setid.mat', dataset_glob_path = "./flowers/*"):
  
  """
    image_labels_path (string): location of imagelabels.mat
    train_test_validation_split_path (string): location of setid.mat
    dataset_glob_path (string): location of dataset with a "/*" at the end for glob.glob to parse
                                                                                                  """
  
  mat = sio.loadmat(train_test_validation_split_path)
  mat = {k:v for k, v in mat.items() if k[0] != '_'}
  data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
  data.to_csv("setid.csv")

  mat = sio.loadmat(image_labels_path)
  mat = {k:v for k, v in mat.items() if k[0] != '_'}
  data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
  data.to_csv("imagelabels.csv")

  df = pd.read_csv("./setid.csv")
  train_ids = np.array(df['trnid'])
  tstids = np.array(df['tstid'])

  train_paths =[]
  for flower in glob.glob(dataset_glob_path):
      if(int(os.path.basename(flower)[6:11]) in train_ids):
          train_paths.append(flower)
        
  test_paths =[]
  for flower in glob.glob(dataset_glob_path):
      if(int(os.path.basename(flower)[6:11]) in tstids):
          test_paths.append(flower)

  image_labels = np.array(pd.read_csv("imagelabels.csv")['labels'])

  train_labels = []
  test_labels = []
  for i in range(len(image_labels)):
      if(i in train_ids):
          train_labels.append(image_labels[i])
      if(i in tstids):
          test_labels.append(image_labels[i])

  train_csv=pd.DataFrame({'labels':train_labels})
  test_csv=pd.DataFrame({'labels':test_labels})



  train = FlowersDataset(train_csv, train_csv, train_paths)
  test = FlowersDataset(test_csv, train_csv, test_paths)

  showImage(test[100][0])
  return train,test
