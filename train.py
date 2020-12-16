from chexpert import loadChexpert
from flowers import loadFlowersDataset
from models import tiny_resnet50, resnet18
from losses import cosine_ohem, CategoricalCrossEntropyLoss

dataset = "chexpert"
#dataset = "flowers"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train = []
test []
train_csv = []
test_csv = []
if(dataset == "chexpert"):
  train, test,train_csv, test_csv = loadChexpert()
elif(dataset=="flowers"):
  train,test, train_csv, test_csv = loadFlowersDataset()
  
batch_size = 100

test_dataloader= DataLoader(test, batch_size = batch_size, shuffle=True)
train_dataloader = DataLoader(train, batch_size = batch_size, shuffle = True)
test_load_all = DataLoader(test, batch_size = len(test), shuffle=True)

class_length = 0

model = resnet18(dataset,train_csv)
model.to(device)

loss_func = "cosine_ohem_0.9ratio_affine_-0.2"

writer = SummaryWriter(log_dir=f"runs/{loss_func}")
print(f"To see tensorboard, run: tensorboard --logdir=runs/{loss_func}")

def create_class_weight(train):
    label_train_arr = [torch.argmax(train[i][1],axis=1) for i in range(len(train))]
    class_weight = torch.sqrt(1.0/(torch.bincount(torch.cat(label_train_arr))))
    class_weight = class_weight/torch.norm(class_weight,2)
    class_weight=class_weight*(1/max(class_weight))
    return class_weight
  
#class_weight = create_class_weight(train)
loss_logger = []
#susbtract y shape by k
criterion = cosine_OHEM(ratio = 0.9,lmbda=-.2,loss_logger=loss_logger)
#CategoricalCrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.00001)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.9)

annealing =True
train_losses = []
test_losses = []
train_correct = []
test_correct = []
softmax_vals = []
specificity_corr = 0
sensitivity_corr = 0
loss = 0
metrics = []
accuracy =[]
epoch = 0

import time
start_time = time.time()

epochs = 70

#change to softmax for ohem type but do log softmax for others
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    specificity_corr = 0
    sensitivity_corr = 0
    
    
    
    for b, (X_train, y_train) in enumerate(train_dataloader):
        b+=1
        #feature selection
        #X_train =(X_train - means) / deviations
        
        y_train_changed =torch.argmax(torch.reshape(y_train.long(),(-1,class_length)),dim=1)
        y_pred = model(X_train)
        y_pred = F.log_softmax(y_pred,dim=1)
        total_size = len(y_train_changed)
        loss,logger = criterion(y_pred, torch.reshape(y_train.long(),(-1,class_length)))
        if(criterion == CategoricalCrossEntropyLoss()):
            loss = torch.mean(loss)
        print(loss)
        softmax_vals.append([y_pred,logger])
        
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train_changed).sum()
        trn_corr += batch_corr
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print interim results     
        sensitivity, specificity,support = sensitivity_specificity_support(np.array(predicted.cpu().detach().numpy()),np.array(y_train_changed.cpu().detach().numpy()),average = "micro")
        specificity_corr+=specificity
        sensitivity_corr+=sensitivity
        metrics.append([specificity, sensitivity, specificity_corr*100*100/(total_size+(batch_size*(b-1))), sensitivity_corr*100*100/(total_size+(batch_size*(b-1)))])
        del y_pred 
        del y_train
        torch.cuda.empty_cache()
        if (b%1==0 and b!=0):
            print(f'epoch: {i:2}  batch: {b:4} [{total_size+(batch_size*(b-1)):6}/{str(len(train))}]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item()*100/(total_size+(batch_size*(b-1))):7.3f}%')
            accuracy.append(trn_corr.item()*100/(total_size+(batch_size*(b-1))))
            print(f'sensitivity: {sensitivity_corr*100*100/(total_size+(batch_size*(b-1)))}%        specificity: {specificity_corr*100*100/(total_size+(batch_size*(b-1)))}%')
            print(f'y_train: {y_train_changed}')
            print(f'predicted: {predicted}')
            epoch+=1
    scheduler.step()
        
        
    train_losses.append(loss)
    train_correct.append(trn_corr)
    
    torch.cuda.empty_cache()
    
    if(i%3==0):
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_dataloader):
                y_test_changed =torch.argmax(torch.reshape(y_test.long(),(-1,class_length)),dim=1)
            
                y_val = model(X_test)
                y_val = F.log_softmax(y_val,dim=1)
            
                predicted = torch.max(y_val.data, 1)[1] 
                tst_corr += (predicted == y_test_changed).sum()
            
        loss,_ = criterion(y_val, torch.reshape(y_test.long(),(-1,class_length)))
        if(criterion == CategoricalCrossEntropyLoss()):
            loss = torch.mean(loss)
        test_losses.append(loss)
        test_correct.append(tst_corr)
        
        #writer.add_scalar('train',torch.tensor(train_losses),i)
        #writer.add_scalar('test',torch.tensor(test_losses),i)

    if((i+1)%5==0):
        model.cpu()
        #previously train_losses was just labeled losses
        torch.save({'epoch':i+1, 'model_state_dict':model.state_dict(), 'hist':softmax_vals, 'loss':loss, 'train_losses':[train_losses,train_correct],'test_losses':[test_losses,test_correct], 'metrics':metrics,'accuracy':accuracy},loss_func+'_'+str(i+1)+'.pt')
        model.cuda()
        
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed    
writer.close()
