checkpoint = "./checkpoints/cosine_ohem_20classes_0.1_50epochs/cosine_OHEM_0.1_50.pt"
checkpoint = torch.load(checkpoint)

dataset = "chexpert"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train = []
test []
train_csv = []
test_csv = []
if(dataset == "chexpert"):
  train, test,train_csv, test_csv = loadChexpert()
elif(dataset=="flowers"):
  train,test, train_csv, test_csv = loadFlowersDataset()

class_length = 0

model = resnet18(dataset,train_csv)
model.load_state_dict(checkpoint['model_state_dict'])

test_load_all = DataLoader(test, batch_size = 100, shuffle=True)

with torch.no_grad():
    correct = 0
    print(1)
    for b,(X_test, y_test) in enumerate(test_load_all):
        
        y_val = model(X_test)
        y_val = F.softmax(y_val,dim=1)
        predicted = torch.max(y_val.data,1)[1]
        correct += (predicted == torch.argmax(torch.reshape(y_test.long(),(-1,class_length)),dim=1)).sum()
        print(correct)
    print((correct/len(test))*100)
