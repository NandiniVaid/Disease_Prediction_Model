import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

df1=pd.read_csv("D:\DP Model\Patients.csv",index_col="ID")
df1.to_csv("D:\DP Model\Patients.csv")

df3=pd.read_csv("D:\DP Model\Diagnosis.csv",index_col="ID")
df3.to_csv("D:\DP Model\Diagnosis.csv")


Password=input("Enter Password:")
if Password=="DisPred":
    while True:
        x1=0
        x2=0
        x3=0
        print('\t\t\t\t    ',"DISEASE  PREDICTION  MODEL",'\t\t\t\t')
        print('\t\t\t\t\t',"BY NANDINI VAID",'\t\t\t\t\t')
        print('\t\t\t',"--------------------MENU--------------------",'\t\t\t')
        print("1.Enter Details.")
        print("2.Predict Disease.")
        print("3.Plot Graphs.")
        print("4.Access Database.")
        print("5.Exit.")
        
        choice=int(input("Enter Choice(1-5):"))


        if choice==1:
            nm=input("Name Of Patient:")
            ag=int(input("Age Of Patient:"))
            gn=input("Gender Of Patient (F/M/Others):")
            dt=input("Date:")
            df2=pd.read_csv("D:\DP Model\Patients.csv")
            s=int((df2.size)/5)
            e=s+1
            l=[e,nm,ag,gn,dt]
            df2.loc[e]=l
            print("Your Details Have Been Entered. Your ID is",e)
            df2.to_csv("D:\DP Model\Patients.csv",index=False)
            df1=pd.read_csv("D:\DP Model\Patients.csv",index_col="ID")



        elif choice==2:

            #List of the symptoms is listed here in list l1.

            l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
            'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
            'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
            'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
            'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
            'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
            'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
            'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
            'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
            'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
            'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
            'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
            'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
            'abnormal_menstruation','dischromic_patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
            'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
            'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
            'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
            'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
            'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
            'yellow_crust_ooze']

            #List of Diseases is listed in list disease.

            disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
            'Peptic ulcer disease','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
            'Migraine','Cervical spondylosis',
            'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
            'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
            'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
            'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
            'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
            'Impetigo']

            l2=[]

            for i in range(0,len(l1)):
                l2.append(0)

            df=pd.read_csv("Prototype.csv")

            #Replacing the values in the imported file by pandas by the inbuilt function replace in pandas.

            df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
            'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
            'Migraine':11,'Cervical spondylosis':12,
            'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
            'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
            'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
            'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
            '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
            'Impetigo':40}},inplace=True)


            X= df[l1]


            y = df[["prognosis"]]
            np.ravel(y)


            #Reading csv named Diseases.csv

            tr=pd.read_csv("Diseases.csv")


            tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
            'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
            'Migraine':11,'Cervical spondylosis':12,
            'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
            'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
            'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
            'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
            '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
            'Impetigo':40}},inplace=True)

            X_test= tr[l1]
            y_test = tr[["prognosis"]]

            np.ravel(y_test)

            def DecisionTree():

                from sklearn import tree

                clf3 = tree.DecisionTreeClassifier() 
                clf3 = clf3.fit(X,y)
    
                # calculating accuracy
                from sklearn.metrics import accuracy_score
                y_pred=clf3.predict(X_test)
    
                psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]

                for k in range(0,len(l1)):
                     for z in psymptoms:
                         if(z==l1[k]):
                            l2[k]=1

                inputtest = [l2]
                predict = clf3.predict(inputtest)
                predicted=predict[0]

                h='no'
                for a1 in range(0,len(disease)):
                    if(predicted == a1):
                        h='yes'
                        break
                if (h=='yes'):
                    print("Prediction1=",disease[a1])
                else:
                    print("Prediction1=","Disease Not Found")
                global x1
                x1=disease[a1]
        
    


            def randomforest():
                from sklearn.ensemble import RandomForestClassifier
                clf4 = RandomForestClassifier()
                clf4 = clf4.fit(X,np.ravel(y))

                # calculating accuracy 
                from sklearn.metrics import accuracy_score
                y_pred=clf4.predict(X_test)
    
                psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]

                for k in range(0,len(l1)):
                    for z in psymptoms:
                        if(z==l1[k]):
                            l2[k]=1

                inputtest = [l2]
                predict = clf4.predict(inputtest)
                predicted=predict[0]

                h='no'
                for a2 in range(0,len(disease)):
                    if(predicted == a2):
                        h='yes'
                        break
                if (h=='yes'):
                    print("Prediction2=",disease[a2])
                else:
                    print("Prediction2=","Disease Not Found")
                global x2
                x2=disease[a2]
    


            def NaiveBayes():
                from sklearn.naive_bayes import GaussianNB
                gnb = GaussianNB()
                gnb=gnb.fit(X,np.ravel(y))

                # calculating accuracy
                from sklearn.metrics import accuracy_score
                y_pred=gnb.predict(X_test)

                psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]
    
                for k in range(0,len(l1)):
                    for z in psymptoms:
                        if(z==l1[k]):
                            l2[k]=1

                inputtest = [l2]
                predict = gnb.predict(inputtest)
                predicted=predict[0]

                h='no'
                for a3 in range(0,len(disease)):
                    if(predicted == a3):
                        h='yes'
                        break
                if (h=='yes'):
                    print("Prediction3=",disease[a3])
                else:
                    print("Prediction3=","Disease Not Found")
                global x3
                x3=disease[a3]
                
        
    
            i1=int(input("Enter ID:"))
            n1=input("Name of the patient:")
            if i1 in list(df1.index):
                Symptom1=input("Symptom1:")
                Symptom2=input("Symptom2:")
                Symptom3=input("Symptom3:")
                Symptom4=input("Symptom4:")
                Symptom5=input("Symptom5:")

                T1=DecisionTree()
                T2=randomforest()
                T3=NaiveBayes()

                
                df4=pd.read_csv("D:\DP Model\Diagnosis.csv")
                l1=[i1,n1,Symptom1,Symptom2,Symptom3,Symptom4,Symptom5,x1,x2,x3]
                df4.loc[i1]=l1
                df4.to_csv("D:\DP Model\Diagnosis.csv",index=False)
                df3=pd.read_csv("D:\DP Model\Diagnosis.csv",index_col="ID")

            else:
                print("PLease Enter Your Details First.")

                

        elif choice==3:
            while True:
                print('\t\t\t',"--------------------Sub-Menu--------------------",'\t\t\t')
                print("1.Plot Graph Depicting Age-Wise Distribution Of Patients.")
                print("2.Plot Graph Depicting Gender-Wise Distribution Of Patients.")
                print("3.Plot Graph Depicting Year-Wise Distribution Of Patients.")
                print("4.Plot Graph Depicting Month-Wise Distribution Of Patients.")
                print("5.Plot Graph Depicting Year-Wise Distribution Of Patients For Different Genders.")
                print("6.Exit.")
                choice1=int(input("Enter Choice(1-6):"))
                
                if choice1==1:
                    while True:
                        # Code to Plot Graph Depicting Age-Wise Distribution Of Patients.
                        print('\t\t\t',"-------------------Graph Choices-------------------",'\t\t\t')
                        print("1.Histogram")
                        print("2.Frequency Polygon")
                        print("3.Move To Other Graphs")
                        choice6=int(input("Enter Choice(1-3):"))
                        if choice6==1:
                            data1=pd.read_csv("D:\DP Model\Patients.csv",usecols=["Age"])
                            df5=pd.DataFrame(data1)
                            df5.plot(kind='hist',bins=10,y="Age",title="Age-Wise Distribution Of Patients",color='Red')
                            plt.xlabel("Age")
                            plt.ylabel("No Of Patients")
                            plt.yticks()
                            plt.show()
                        elif choice6==2:
                            data1=pd.read_csv("D:\DP Model\Patients.csv",usecols=["Age"])
                            df5=pd.DataFrame(data1)
                            arr1=np.array(df5["Age"])
                            y,edges=np.histogram(arr1)
                            mid=0.5*(edges[1:]+edges[:-1])
                            df5.plot(kind='hist',y="Age",title="Age-Wise Distribution Of Patients",color='yellow',legend=None)
                            plt.plot(mid,y,'-D',color="green")
                            plt.xlabel("Age")
                            plt.ylabel("No Of Patients")
                            plt.yticks()
                            plt.show() 
                        elif choice6==3:
                            break
                        else:
                           print("Please Enter A Valid Choice.") 
                            
                    
                elif choice1==2:
                    while True:
                        # Code to Plot Graph Depicting Gender-Wise Distribution Of Patients.
                        print('\t\t\t',"-------------------Graph Choices-------------------",'\t\t\t')
                        print("1.Pie Chart")
                        print("2.Line Chart")
                        print("3.Scatter Chart")
                        print("4.Bar Chart")
                        print("5.Move To Other Graphs")
                        choice5=int(input("Enter Choice(1-5):"))
                        if choice5==1:
                            sf=0
                            sm=0
                            so=0
                            list1=list(df1.Gender)
                            for i in range(len(list1)):
                                if list1[i]=="F":
                                    sf=sf+1
                                elif list1[i]=="M":
                                    sm=sm+1
                                else:
                                    so=so+1
                            df6=pd.DataFrame({"No.Of Patients":[sf,sm,so]},index=["Female","Male","Others"])
                            df6.plot(kind="pie",y="No.Of Patients",title="Gender-Wise Distribution Of Patients",autopct="%.2f")
                            plt.show()
                        elif choice5==2:
                            sf=0
                            sm=0
                            so=0
                            list1=list(df1.Gender)
                            for i in range(len(list1)):
                                if list1[i]=="F":
                                    sf=sf+1
                                elif list1[i]=="M":
                                    sm=sm+1
                                else:
                                    so=so+1
                            df6=pd.DataFrame({"No.Of Patients":[sf,sm,so]},index=["Female","Male","Others"])
                            df6.plot(kind="line",y="No.Of Patients",title="Gender-Wise Distribution Of Patients",color="black",linestyle="--")
                            plt.xlabel("Gender")
                            plt.ylabel("No.Of Patients")
                            plt.yticks([sf,sm,so])
                            plt.show()
                        elif choice5==3:
                            sf=0
                            sm=0
                            so=0
                            list1=list(df1.Gender)
                            for i in range(len(list1)):
                                if list1[i]=="F":
                                    sf=sf+1
                                elif list1[i]=="M":
                                    sm=sm+1
                                else:
                                    so=so+1
                            df6=pd.DataFrame({"No.Of Patients":[sf,sm,so]},index=["Female","Male","Others"])
                            plt.scatter(x=["Female","Male","Others"],y=[sf,sm,so],color="red",marker="P",s=70)
                            plt.title("Gender-Wise Distribution Of Patients")
                            plt.xlabel("Gender")
                            plt.ylabel("No.Of Patients")
                            plt.yticks([sf,sm,so])
                            plt.show()
                        elif choice5==4:
                            sf=0
                            sm=0
                            so=0
                            list1=list(df1.Gender)
                            for i in range(len(list1)):
                                if list1[i]=="F":
                                    sf=sf+1
                                elif list1[i]=="M":
                                    sm=sm+1
                                else:
                                    so=so+1
                            df6=pd.DataFrame({"No.Of Patients":[sf,sm,so]},index=["Female","Male","Others"])
                            df6.plot(kind="bar",y="No.Of Patients",title="Gender-Wise Distribution Of Patients",color=["red","yellow","blue"],legend=None)
                            plt.xlabel("Gender")
                            plt.ylabel("No.Of Patients")
                            plt.yticks([sf,sm,so])
                            plt.show()
                        elif choice5==5:
                            break
                        else:
                           print("Please Enter A Valid Choice.") 
                            
                    
                elif choice1==3:
                    while True:
                        # Code to Plot Graph Depicting Year-Wise Distribution Of Patients.
                        print('\t\t\t',"-------------------Graph Choices-------------------",'\t\t\t')
                        print("1.Pie Chart")
                        print("2.Line Chart")
                        print("3.Scatter Chart")
                        print("4.Bar Chart")
                        print("5.Move To Other Graphs")
                        choice3=int(input("Enter Choice(1-5):"))
                        if choice3==1:
                            list2=list(df1.Date)
                            df7=pd.DataFrame(columns=["Year"])
                            for i in range(len(list2)):
                                date=list2[i]
                                datem=datetime.datetime.strptime(date,"%Y-%m-%d")
                                df7.loc[i]=datem.year
                            df8=df7.groupby("Year")
                            df9=df8.size()
                            df10=pd.DataFrame({"No.Of Patients":list(df9.values)},index=list(df9.index))
                            df10.plot(kind="pie",y="No.Of Patients",title="Year-Wise Distribution Of Patients",autopct="%.2f")
                            plt.show()
                        elif choice3==2:
                            list2=list(df1.Date)
                            df7=pd.DataFrame(columns=["Year"])
                            for i in range(len(list2)):
                                date=list2[i]
                                datem=datetime.datetime.strptime(date,"%Y-%m-%d")
                                df7.loc[i]=datem.year
                            df8=df7.groupby("Year")
                            df9=df8.size()
                            df10=pd.DataFrame({"No.Of Patients":list(df9.values)},index=list(df9.index))
                            df10.plot(kind="line",y="No.Of Patients",title="Year-Wise Distribution Of Patients", color="blue")
                            plt.xlabel("Year")
                            plt.ylabel("No.Of Patients")
                            plt.xticks(list(df9.index))
                            plt.yticks(list(df9.values))
                            plt.show()
                        elif choice3==3:
                            list2=list(df1.Date)
                            df7=pd.DataFrame(columns=["Year"])
                            for i in range(len(list2)):
                                date=list2[i]
                                datem=datetime.datetime.strptime(date,"%Y-%m-%d")
                                df7.loc[i]=datem.year
                            df8=df7.groupby("Year")
                            df9=df8.size()
                            df10=pd.DataFrame({"No.Of Patients":list(df9.values)},index=list(df9.index))
                            plt.scatter(x=list(df9.index),y=list(df9.values),color="blue")
                            plt.title("Year-Wise Distribution Of Patients")
                            plt.xlabel("Year")
                            plt.ylabel("No.Of Patients")
                            plt.xticks(list(df9.index))
                            plt.yticks(list(df9.values))
                            plt.show()
                        elif choice3==4:
                            list2=list(df1.Date)
                            df7=pd.DataFrame(columns=["Year"])
                            for i in range(len(list2)):
                                date=list2[i]
                                datem=datetime.datetime.strptime(date,"%Y-%m-%d")
                                df7.loc[i]=datem.year
                            df8=df7.groupby("Year")
                            df9=df8.size()
                            df15=pd.DataFrame({"Year":list(df9.index),"No.Of Patients":list(df9.values)})
                            plt.bar(x=df15.Year,height=list(df9.values),color="cyan")
                            plt.title("Year-Wise Distribution Of Patients")
                            plt.xlabel("Year")
                            plt.ylabel("No.Of Patients")
                            plt.xticks(list(df9.index))
                            plt.yticks(list(df9.values))
                            plt.show()
                        elif choice3==5:
                            break
                        else:
                            print("Please Enter A Valid Choice.")
                            
                    
                elif choice1==4:
                    while True:
                        # Code to Plot Graph Depicting Month-Wise Distribution Of Patients
                        print('\t\t\t',"-------------------Graph Choices-------------------",'\t\t\t')
                        print("1.Pie Chart")
                        print("2.Line Chart")
                        print("3.Scatter Chart")
                        print("4.Bar Chart")
                        print("5.Move To Other Graphs")
                        choice4=int(input("Enter Choice(1-5):"))
                        if choice4==1:
                            list3=list(df1.Date)
                            df11=pd.DataFrame(columns=["Month"])
                            for i in range(len(list3)):
                                date=list3[i]
                                datem=datetime.datetime.strptime(date,"%Y-%m-%d")
                                month_num=str(datem.month)
                                var1=datetime.datetime.strptime(month_num,"%m")
                                df11.loc[i]=var1.strftime("%B")
                            df12=df11.groupby("Month")
                            df13=df12.size()
                            df14=pd.DataFrame({"No.Of Patients":list(df13.values)},index=list(df13.index))
                            df14.plot(kind="pie",y="No.Of Patients",title="Month-Wise Distribution Of Patients",autopct="%.2f")
                            plt.show()
                        elif choice4==2:
                            list3=list(df1.Date)
                            df11=pd.DataFrame(columns=["Month"])
                            for i in range(len(list3)):
                                date=list3[i]
                                datem=datetime.datetime.strptime(date,"%Y-%m-%d")
                                month_num=str(datem.month)
                                var1=datetime.datetime.strptime(month_num,"%m")
                                df11.loc[i]=var1.strftime("%B")
                            df12=df11.groupby("Month")
                            df13=df12.size()
                            df14=pd.DataFrame({"No.Of Patients":list(df13.values)},index=list(df13.index))
                            df14.plot(kind="line",y="No.Of Patients",title="Month-Wise Distribution Of Patients",color="green")
                            plt.xlabel("Month")
                            plt.ylabel("No.Of Patients")
                            plt.yticks(list(df13.values))
                            plt.show()
                        elif choice4==3:
                            list3=list(df1.Date)
                            df11=pd.DataFrame(columns=["Month"])
                            for i in range(len(list3)):
                                date=list3[i]
                                datem=datetime.datetime.strptime(date,"%Y-%m-%d")
                                month_num=str(datem.month)
                                var1=datetime.datetime.strptime(month_num,"%m")
                                df11.loc[i]=var1.strftime("%B")
                            df12=df11.groupby("Month")
                            df13=df12.size()
                            df14=pd.DataFrame({"No.Of Patients":list(df13.values)},index=list(df13.index))
                            plt.scatter(x=list(df13.index),y=list(df13.values),color="purple",marker="*")
                            plt.title("Month-Wise Distribution Of Patients")
                            plt.xlabel("Month")
                            plt.ylabel("No.Of Patients")
                            plt.yticks(list(df13.values))
                            plt.show()
                        elif choice4==4:
                            list3=list(df1.Date)
                            df11=pd.DataFrame(columns=["Month"])
                            for i in range(len(list3)):
                                date=list3[i]
                                datem=datetime.datetime.strptime(date,"%Y-%m-%d")
                                month_num=str(datem.month)
                                var1=datetime.datetime.strptime(month_num,"%m")
                                df11.loc[i]=var1.strftime("%B")
                            df12=df11.groupby("Month")
                            df13=df12.size()
                            df16=pd.DataFrame({"Month":list(df13.index),"No.Of Patients":list(df13.values)})
                            plt.bar(x=df16.Month,height=list(df13.values),color="magenta")
                            plt.title("Month-Wise Distribution Of Patients")
                            plt.xlabel("Month")
                            plt.ylabel("No.Of Patients")
                            plt.yticks(list(df13.values))
                            plt.show()
                        elif choice4==5:
                            break
                        else:
                            print("Please Enter A Valid Choice.")
                            
                    
                elif choice1==5:
                    while True:
                        # Code to Plot Graph Depicting Year-Wise Distribution Of Patients For Different Genders .
                        print('\t\t\t',"-------------------Graph Choices-------------------",'\t\t\t')
                        print("1.Bar Chart")
                        print("2.Move To Other Graphs")
                        choice7=int(input("Enter Choice(1/2):"))
                        if choice7==1:
                            list4=list(df1.Date)
                            df17=pd.DataFrame(columns=["Gender","Year"])
                            df17["Gender"]=list(df1.Gender)
                            for i in range(len(list4)):
                                date=list4[i]
                                datem=datetime.datetime.strptime(date,"%Y-%m-%d")
                                df17.loc[i,"Year"]=datem.year
                            df18=pd.DataFrame(columns=["Year"])
                            df21=pd.DataFrame(columns=["Year","No.Of Female Patients"])
                            df22=pd.DataFrame(columns=["Year"])
                            df25=pd.DataFrame(columns=["Year","No.Of Male Patients"])
                            df26=pd.DataFrame(columns=["Year"])
                            df29=pd.DataFrame(columns=["Year","No.Of Other Patients"])
                            for b in range(int((df17.size)/2)):
                                if df17.loc[b,"Gender"]=="F":
                                    mf=df18.size
                                    df18.loc[mf]=df17.loc[b,"Year"]                                    
                                elif df17.loc[b,"Gender"]=="M":
                                    mm=df22.size
                                    df22.loc[mm]=df17.loc[b,"Year"]
                                else:
                                    mo=df26.size
                                    df26.loc[mo]=df17.loc[b,"Year"]
                            
                            df19=df18.groupby("Year")
                            df20=df19.size()
                            df21.loc[:,"Year"]=list(df20.index)
                            df21.loc[:,"No.Of Female Patients"]=list(df20.values)

                            df23=df22.groupby("Year")
                            df24=df23.size()
                            df25.loc[:,"Year"]=list(df24.index)
                            df25.loc[:,"No.Of Male Patients"]=list(df24.values)

                            df27=df26.groupby("Year")
                            df28=df27.size()
                            df29.loc[:,"Year"]=list(df28.index)
                            df29.loc[:,"No.Of Other Patients"]=list(df28.values)


                            df31=df17.groupby("Year")
                            df32=df31.size()
                            df30=pd.DataFrame(index=list(df32.index),columns=["No.Of Female Patients","No.Of Male Patients","No.Of Other Patients"])
                            df30.loc[list(df20.index),"No.Of Female Patients"]=list(df20.values)
                            df30.loc[list(df24.index),"No.Of Male Patients"]=list(df24.values)
                            df30.loc[list(df28.index),"No.Of Other Patients"]=list(df28.values)
                            df30.fillna(0)
                            df33=pd.DataFrame({"Year":list(df30.index),"No.Of Female Patients":list(df30["No.Of Female Patients"]),"No.Of Male Patients":list(df30["No.Of Male Patients"]),"No.Of Other Patients":list(df30["No.Of Other Patients"])})
                            df33.plot(kind="bar",x="Year",color=["red","purple","yellow"])
                            plt.title("Year-Wise Distribution Of Patients For Different Genders")
                            plt.yticks(list(df32.values))
                            plt.xlabel("Year")
                            plt.ylabel("No.Of Patients")
                            plt.show()
                        elif choice7==2:
                            break
                        else:
                            print("Please Enter A Valid Choice.") 


                elif choice1==6:
                    break

                
                else:
                    print("Please Enter A Valid Choice.")



        elif choice==4:
            print("! Please confirm if you are an admin !")
            Password1=input("Enter Admin Password:")
            if Password1=="AdminPass":
                while True:
                    print('\t\t\t',"--------------------Sub-Menu--------------------",'\t\t\t')
                    print("1.Display Patient Database")
                    print("2.Display Diagnosis Database")
                    print("3.Exit")
                    choice2=int(input("Enter Choice(1-3):"))
                    if choice2==1:
                        print('\t\t\t\t\t',"PATIENT DATABASE",'\t\t\t\t\t')
                        print(df1)
                    elif choice2==2:
                        print('\t\t\t\t\t',"DIAGNOSIS DATABASE",'\t\t\t\t\t')
                        print(df3)
                    elif choice2==3:
                        break
                    else:
                        print("Please Enter A Valid Choice.")

            else:
                print("Password Incorrect.")



        elif choice==5:
            break



        else:
            print("Please Enter A Valid Choice.")


else:
    print("Password Incorrect.")
