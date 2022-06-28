#2022 Makkiblog.com MIT License
#

from win32com.client import Dispatch
import datetime as date
import os

namefolder = "FOLDERNAME" #Savefile path
save_path = r'.\hogehoge' #Path name
sub_ = 'hoge' #Subject name
rollback = -1 #0 for today, -1,-2,-3 for one, two, three days before

lag = 1 

outlook = Dispatch("Outlook.Application").GetNamespace("MAPI")
inbox = outlook.GetDefaultFolder("6") #Received Tray
MAIL = inbox.Folders("namefolder")  
Proto = MAIL.Items
val_date = (date.date.today() - date.timedelta(lag)).strftime("%y%m%d")


for msg in Proto:
    #print(msg) 
    rec = (int(msg.SentOn.strftime("%y%m%d")) - int(val_date))
    #print(msg.SentOn.strftime("%y%m%d"))
    print(msg.subject)
    print(rec)
    if rec > rollback:  
        if msg.subject.startswith(sub_):
            text = msg.subject
            subject = '_'.join(text.split())
            subject = subject.replace('RE:','')
            subject = subject.replace(':','')
            subject = subject.replace('/:','')
            dirName = save_path + "/"  + subject
            os.makedirs(dirName, exist_ok=True)
            for att in msg.Attachments:     
               att.SaveAsFile(dirName + "/" + att.FileName)

#    elif msg.SentOn.strftime("%d/%m/%y") < val_date:
#        break


#for msg in all_inbox:
#    print(msg)
#
#    if sub_ in msg.Subject:
#        for att in msg.Attachments:
#            if att_ in att.FileName:
#                att.SaveAsFile(save_path + "/" + att.FileName)
#
#    elif msg.SentOn.strftime("%d/%m/%y") < val_date:
#        break
