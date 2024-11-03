Anthropic has at least 3 capabilities currently available in their computer use release:
    – Drive a web browser
    – Drive bash execution of command line programs
    – Drive file editing

Looking at previous conversations, it appears that bash execution and file editing have not yet been explored much by our group, so I decided to look into bash execution first.

Some projects like Open Interpreter and Agent.exe already support the computer-use API, so we may see a bunch of such interfaces pop up soon.
However, with them you have less configurability than a home-grown option, so I decided to explore the API myself.

A program is attached that demos the bash tools, and below are some directions of how to run the program. There are examples for use of trivial bash commands as well as more interesting use, e.g. using it to add data to a sqlite3 database, and query it. (It should be noted: while bash and file editing are done here via text command line, that text could instead be supplied using a voice assistant interface.)

The script as written calls Claude in a loop until it decides the task is complete. Typically, this ends up with not just the desired command being generated and executed, but also a verification step where Claude ensures that everything worked out correctly. If you just wanted the command to run without potentially lengthy verification steps, you could just call Claude once.

Now, that I have these working examples for bash, I will examine file editing next.


Program directions:

pip install -U anthropic  # Do this first

––––  Simple requests for bash command execution

touch temp
python work.py "List all files in this directory, and if there is one named temp, rename it to tempXXX"

mkdir -p dird1/dird2/dird3
touch dird1/dird2/dird3/halloween.py
python work.py "Find a file named halloween.py here or below"

––––  More complex requests (sqlite3 commands)

# You need sqlite3 installed for this one
python work.py “Create an sqlite3 database named foobar.db, then create a USER table with two fields: uname age"
sqlite3 foobar.db .table

python work.py “Put two entries into the foobar.db table named USER; uname bob age 22 and sam age 33"
    # Note that anthropic verifies the inserts by doing a SELECT

sqlite3 foobar.db "select * from USER"  # Or you could do the select yourself
