#!/home/simon/anaconda3/envs/tensorflow-gpu/bin/python
import tarfile
import getpass
import paramiko
import os
from Utils.Data import getListOfFiles
HOST = "deeprain.ddns.net"
PORT = 22
DOWNLOAD_PATH = "/home/bananapi/Dataset/DeepRain"

def printProgress(transferred, toBeTransferred):

    width = 50
    percent = transferred / toBeTransferred
    size = int(percent * width)
    fetched = transferred / (1024**2)
    total = toBeTransferred / (1024**2)
    progress = "%(current)d/%(total)d Mb"

    args = {
        "current": fetched,
        "total": total
    }
    bar = progress + ' [' + '#' * size + '#' + '.'*(width - size) + ']'
    print("\r"+bar % args, end='')



def downloadDataSet(root_dir,host,port,DOWNLOAD_PATH,username, pswd):

    path = root_dir

    transport = paramiko.Transport((host, port))


    transport.connect(username=username, password=pswd)
    sftp = paramiko.SFTPClient.from_transport(transport)

    if os.path.exists(path) == False:
        os.makedirs(path)

    sftp.get(remotepath=DOWNLOAD_PATH, localpath=path +
                 "/"+DOWNLOAD_PATH.split("/")[-1], callback=printProgress)
    print("\n")

def extract(root_dir,tarfilename):

    tf = tarfile.open(root_dir+"/"+tarfilename, 'r')
    tf.extractall(path=root_dir)

    os.remove(root_dir+"/"+tarfilename)

def getDataSet(working_dir,year=None,username=None,pswd=None):
    prefix = "YW2017.002_"
    suffix = ".tar.gz"
    start = 2008
    end   = 2017
    filenames = []
    removeyears = []

    try:
        
        # check if files exists
        files = getListOfFiles(working_dir)
        for y in year:
            checkFile = prefix + str(y)
            for file in files:

                if checkFile in file:
                    print("\u001b[33mFound Year \u001b[0m: ",y,\
                        "=> won't download this year again... please check for consistency")
                    removeyears.append(y)
                    break


        for y in removeyears:
            year.remove(y)

    except Exception as e:
        print("No Data Found\n")

    if year is None:
        filenames = [prefix + str(i) + suffix for i in range(start,end+1)]
    else:
        if type(year) is not list:
            print("[Error] : parameter year needs to be a list of int")
            exit(-1)

        for y in year:
            if len(str(y)) == 4 and str(y).isdigit():
                filenames.append(prefix + str(y) + suffix)
            else:
                print("[WARNING] : "+str(y)+" wrong format YYYY")




    if len(year) > 0 :
        print("Downloading from:", str(HOST))
        if username is None:
            username = input('Enter your username\n')
        if pswd is None:
            pswd = getpass.getpass('Enter your password:\n')

        print("Connecting to "+HOST+"...")

        for filename in filenames:
            print("Downloading ",filename,"...")
            downloadDataSet(working_dir,HOST,PORT,os.path.join(DOWNLOAD_PATH,filename),username,pswd)

        for filename in filenames:
            print("Extracting ",filename,"...")
            extract(working_dir,filename)

    print("\u001b[32mFinished Loading Dataset\n \u001b[0m")

