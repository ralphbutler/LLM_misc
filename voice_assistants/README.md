# example voice assistant

There are commercial packages that can provide a more sophisticated implementation, e.g.:

*    elevenlabs
*    assemblyai
*    whisper (small demos in extra_demos sub-dir)
*    faster-whisper (small demos in extra_demos sub-dir)
*    etc

But I chose to use regular python modules for now.  Once the user's voice is converted
to text, I currently use GPT-4 to provide a solution, but we can swap to any open LLM
that produces good structured output.  The LLM is used to figure out what the user is
requesting and turn it into a valid terminal (unix) command.

Random scraps of code using some of the above packages are in the sub-dir named extra_code.

I may decide to do a better version later.

Note the current wake phrase is "hello there", easily alterable at top of program.
That phrase is required prior to giving each command.
For example, I do something like:

*     Isay:  hello there
*     (pause until it states an offer to help; time can vary esp at first)
*     Isay:  when was the last date that the system was booted
*     (it should show this command in the terminal "who -b")

You may want to change some of the time-out variables.
