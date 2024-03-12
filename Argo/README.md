# Using Argo and LangChain from off-site
## 1. In terminal window 1:
        ssh -D 32000 -CqN butlerr@homes.cels.anl.gov
            this will require you to enter your passphrase twice
            and the process will hang until killed with ^C

## 2. In terminal window 2 (to run pgms and debug):
        export https_proxy=socks5h://localhost:32000

## 3. python3 demo1.py
## 4. python3 demo2.py

#### If the two demo programs run fine, you should be good to go.
