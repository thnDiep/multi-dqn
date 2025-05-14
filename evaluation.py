import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def full_ensemble(df):
    """Tính toán ensemble với 100% đồng thuận"""
    m1 = df.eq(1).all(axis=1)
    m2 = df.eq(2).all(axis=1)
    local_df = df.copy()
    local_df['ensemble'] = np.select([m1, m2], [1, -1], 0)
    local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)
    return local_df

def perc_ensemble(df, thr=0.7):
    """Tính toán ensemble với ngưỡng đồng thuận nhất định"""
    c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
    c2 = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
    return pd.DataFrame(np.select([c1, c2], [1, -1], 0), index=df.index, columns=['ensemble'])

def generate_ensemble_decisions(num_walks, ensemble_folder, result_file):
    """Tạo quyết định ensemble từ các walk khác nhau"""
    fulldf = None
    
    for j in range(num_walks):
        df = pd.read_csv(f"{ensemble_folder}/walk{j}ensemble_test.csv", index_col='Date')
        if fulldf is None:
            fulldf = full_ensemble(df)
        else:
            fulldf = fulldf.append(full_ensemble(df))
    
    fulldf.index = pd.to_datetime(fulldf.index)
    fulldf.to_csv(result_file)

def ensemble(numWalks, perc, type, numDel, market, ensemble_folder):
    """
    Tính toán các metrics cho ensemble
    
    Args:
        numWalks: Số lượng walk
        perc: Ngưỡng đồng thuận (0: 100%, 0.9: 90%, etc)
        type: Loại dữ liệu ("valid" hoặc "test")
        numDel: Số iteration cần xóa
        market: Tên thị trường (default: "dax")
    """
    dollSum=0
    rewSum=0
    posSum=0
    negSum=0
    covSum=0
    numSum=0

    values=[]
    columns = ["Iteration","Reward%","#Wins","#Losses","Dollars","Coverage","Accuracy"]
    
    # Đọc dữ liệu thị trường với tên thị trường được truyền vào
    market_data = pd.read_csv(f"./datasets/{market}Day.csv", index_col='Date')
    
    for j in range(0, numWalks):
        df = pd.read_csv(f"{ensemble_folder}/walk{j}ensemble_{type}.csv", index_col='Date')

        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]
        
        if perc==0:
            df=full_ensemble(df)
        else:
            df=perc_ensemble(df,perc)

        num=0
        rew=0
        pos=0
        neg=0
        doll=0
        cov=0
        for date, i in df.iterrows():
            num+=1

            if date in market_data.index:
                if (i['ensemble']==1):
                    rew+= (market_data.at[date,'Close']-market_data.at[date,'Open'])/market_data.at[date,'Open']
                    pos+= 1 if rew > 0 else 0
                    neg+= 0 if rew > 0 else 1
                    doll+=(market_data.at[date,'Close']-market_data.at[date,'Open'])*50
                    cov+=1
                elif (i['ensemble']==2):
                    rew+=-(market_data.at[date,'Close']-market_data.at[date,'Open'])/market_data.at[date,'Open']
                    neg+= 0 if -rew > 0 else 1
                    pos+= 1 if -rew > 0 else 0
                    cov+=1
                    doll+=-(market_data.at[date,'Close']-market_data.at[date,'Open'])*50
        
        values.append([str(round(j,2)),str(round(rew,2)),str(round(pos,2)),str(round(neg,2)),str(round(doll,2)),str(round(cov/num,2)),(str(round(pos/cov,2)) if (cov>0) else "")])
        
        dollSum+=doll
        rewSum+=rew
        posSum+=pos
        negSum+=neg
        covSum+=cov
        numSum+=num

    values.append(["sum",
                   str(round(rewSum,2)),
                   str(round(posSum,2)),
                   str(round(negSum,2)),
                   str(round(dollSum,2)),
                   str(round(covSum/numSum,2)),
                   (str(round(posSum/covSum,2)) if (covSum>0) else "")])
    return values, columns

def plot_training_metrics(num_walks, num_epochs, walk_files, result_file):
    """Vẽ các biểu đồ metrics trong quá trình training"""
    pdf = PdfPages(result_file)
    num_plots = 11
    plt.figure(figsize=((num_epochs/10)*(num_walks+1),num_plots*5))
    
    for i in range(1, num_walks+1):
        document = pd.read_csv(f"{walk_files}{i}.csv")
        
        # Accuracy
        plt.subplot(num_plots, num_walks, 0*num_walks + i)
        plt.plot(document['Iteration'].tolist(), document['trainAccuracy'].tolist(), 'g', label='Train')
        plt.plot(document['Iteration'].tolist(), document['validationAccuracy'].tolist(), 'g', label='Validation')
        plt.xticks(range(0, num_epochs, 4))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.ylim(-0.05, 1.05)
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title(f'Walk {i}\n\nAccuracy')

        # Coverage
        plt.subplot(num_plots, num_walks, 1*num_walks + i)
        plt.plot(document['Iteration'].tolist(), document['trainCoverage'].tolist(), 'g', label='Train')
        plt.plot(document['Iteration'].tolist(), document['validationCoverage'].tolist(), 'g', label='Validation')
        plt.xticks(range(0, num_epochs, 4))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.ylim(-0.05, 1.05)
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title('Coverage')

        # Train Reward
        plt.subplot(num_plots, num_walks, 2*num_walks + i)
        plt.plot(document['Iteration'].tolist(), document['trainReward'].tolist(), 'b', label='Train')
        plt.xticks(range(0, num_epochs, 4))
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title('Train Reward')

        # Validation Reward
        plt.subplot(num_plots, num_walks, 3*num_walks + i)
        plt.plot(document['Iteration'].tolist(), document['validationReward'].tolist(), 'g', label='Validation')
        plt.xticks(range(0, num_epochs, 4))
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title('Validation Reward')

        # Long %
        plt.subplot(num_plots, num_walks, 4*num_walks + i)
        plt.plot(document['Iteration'].tolist(), document['trainLong%'].tolist(), 'b', label='Train')
        plt.plot(document['Iteration'].tolist(), document['validationLong%'].tolist(), 'g', label='Validation')
        plt.xticks(range(0, num_epochs, 4))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.ylim(-0.05, 1.05)
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title('Long %')

        # Short %
        plt.subplot(num_plots, num_walks, 5*num_walks + i)
        plt.plot(document['Iteration'].tolist(), document['trainShort%'].tolist(), 'b', label='Train')
        plt.plot(document['Iteration'].tolist(), document['validationShort%'].tolist(), 'g', label='Validation')
        plt.xticks(range(0, num_epochs, 4))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.ylim(-0.05, 1.05)
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title('Short %')

        # Hold %
        plt.subplot(num_plots, num_walks, 6*num_walks + i)
        plt.plot(document['Iteration'].tolist(), [1-x for x in document['trainCoverage'].tolist()], 'b', label='Train')
        plt.plot(document['Iteration'].tolist(), [1-x for x in document['validationCoverage'].tolist()], 'g', label='Validation')
        plt.xticks(range(0, num_epochs, 4))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.ylim(-0.05, 1.05)
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title('Hold %')

        # Long Accuracy
        plt.subplot(num_plots, num_walks, 7*num_walks + i)
        plt.plot(document['Iteration'].tolist(), document['trainLongAcc'].tolist(), 'b', label='Train')
        plt.plot(document['Iteration'].tolist(), document['validationLongAcc'].tolist(), 'g', label='Validation')
        plt.xticks(range(0, num_epochs, 4))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.ylim(-0.05, 1.05)
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title('Long Accuracy')

        # Short Accuracy
        plt.subplot(num_plots, num_walks, 8*num_walks + i)
        plt.plot(document['Iteration'].tolist(), document['trainShortAcc'].tolist(), 'b', label='Train')
        plt.plot(document['Iteration'].tolist(), document['validationShortAcc'].tolist(), 'g', label='Validation')
        plt.xticks(range(0, num_epochs, 4))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.ylim(-0.05, 1.05)
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title('Short Accuracy')

        # Long Precision
        plt.subplot(num_plots, num_walks, 9*num_walks + i)
        plt.plot(document['Iteration'].tolist(), document['trainLongPrec'].tolist(), 'b', label='Train')
        plt.plot(document['Iteration'].tolist(), document['validLongPrec'].tolist(), 'g', label='Validation')
        plt.xticks(range(0, num_epochs, 4))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.ylim(-0.05, 1.05)
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title('Long Precision')

        # Short Precision
        plt.subplot(num_plots, num_walks, 10*num_walks + i)
        plt.plot(document['Iteration'].tolist(), document['trainShortPrec'].tolist(), 'b', label='Train')
        plt.plot(document['Iteration'].tolist(), document['validShortPrec'].tolist(), 'g', label='Validation')
        plt.xticks(range(0, num_epochs, 4))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.ylim(-0.05, 1.05)
        plt.axhline(y=0, color='k', linestyle='-')
        plt.legend()
        plt.grid()
        plt.title('Short Precision')

    plt.suptitle("Esperimento SP500 5 (Only long):\n"
            +"Target model update: 1e-1\n"
            +"Model: 35 neurons single layer\n"
            +"Memory-Window Length: 10000-1\n"
            +"Train length: 5 Years\n"
            +"Validation length: 6 Months\n"
            +"Test lenght: 6 Months\n"
            +"Starting period: 2010-01-01\n"
            +"Other changes: Does only Long actions"
            ,size=30
            ,weight=3
            ,ha='left'
            ,x=0.1
            ,y=0.99)
    pdf.savefig()
    pdf.close()

def plot_ensemble_results(num_walks, market, ensemble_folder, result_file):
    """Vẽ các bảng kết quả ensemble với các ngưỡng khác nhau"""
    pdf = PdfPages(result_file)
    
    thresholds = [0, 0.9, 0.8, 0.7, 0.6]
    titles = ["FULL ENSEMBLE", "90% ENSEMBLE", "80% ENSEMBLE", "70% ENSEMBLE", "60% ENSEMBLE"]
    
    for threshold, title in zip(thresholds, titles):
        plt.figure(figsize=(10, 5))
        
        # Validation results
        plt.subplot(1, 2, 1)
        plt.axis('off')
        val, col = ensemble(num_walks, threshold, "valid", 0, market, ensemble_folder)
        t = plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(6)
        plt.title("Valid")
        
        # Test results
        plt.subplot(1, 2, 2)
        plt.axis('off')
        val, col = ensemble(num_walks, threshold, "test", 0, market, ensemble_folder)
        t = plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(6)
        plt.title("Test")
        
        plt.suptitle(title)
        pdf.savefig()
    
    pdf.close()

def combine_signals():
    """Kết hợp các tín hiệu từ long và short để tạo quyết định cuối cùng"""
    long = [[],[]]
    short = [[],[]]

    longs = pd.read_csv("./Output/results/spLong.csv")
    shorts = pd.read_csv("./Output/results/spShort.csv")

    long[0] = longs.ix[:,"Date"].tolist()
    long[1] = longs.ix[:,"ensemble"].tolist()
    short[0] = shorts.ix[:,"Date"].tolist()
    short[1] = shorts.ix[:,"ensemble"].tolist()

    # Tạo thư mục results nếu chưa tồn tại
    os.makedirs("./Output/results", exist_ok=True)
    
    output = open("./Output/results/finalEnsemble.csv", "w+")
    output.write("date,ensemble\n")

    for i in range(0,len(long[0])):
        if(long[0][i]==short[0][i]):
            output.write(str(long[0][i]) + "," + str(long[1][i]+short[1][i]) + "\n")
    
    output.close()

def evaluate_model(num_walks, num_epochs, market, walk_files, ensemble_folder, result_file):
    """Hàm chính để đánh giá và trực quan hóa kết quả"""
    # Vẽ biểu đồ ensemble results
    plot_ensemble_results(
        num_walks=num_walks,
        market=market,
        ensemble_folder=ensemble_folder,
        result_file=f"{result_file}_ensemble.pdf"
    )

    plot_training_metrics(num_walks=num_walks, 
                          num_epochs=num_epochs, 
                          walk_files=walk_files,
                          result_file=f"{result_file}_training.pdf")