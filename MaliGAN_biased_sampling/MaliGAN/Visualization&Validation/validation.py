from Bio.SeqUtils.ProtParam import ProteinAnalysis
def validation(filepath):
    amp =[]
    Total_AMP = 0
    Non_AMP = 0
    AMP = 0
    with open(filepath,'r') as fin:
        for _ in fin:
            amp.append(_)

    for i in amp:
        X = ProteinAnalysis(i)
        Nc = X.charge_at_pH(pH=7)
        Ip = X.isoelectric_point()
        if Nc>=2 and Ip>=5:
            Total_AMP+=1
            AMP+=1
            #print('AMP:',i)
        else:
            Total_AMP+=1
            Non_AMP+=1
            #print('NonAmp:',i)
    print('Total_AMP:{} Non_AMP:{} AMP:{}'.format(Total_AMP,Non_AMP,AMP))


validation('AMP_Sequence.txt')
validation('amp_generated.txt')














































# for sentence in amp:
#     print(sentence)
    # for letter in sentence:
    #     for key,value in charge_value.items():
    #         if str(letter) == key:
    #             charge += value
    #             print(charge)
    #             #print(key)

