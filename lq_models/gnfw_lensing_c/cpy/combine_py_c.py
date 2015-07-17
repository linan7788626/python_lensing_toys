from numpy import *;
import pexpect;
import string;

executable_filename = "data_cruncher";

data = arange(15); # this is a numpy array, we need to convert it to list for normal python usage
data_list = data.tolist();
data_str = string.join(str(x) for x in data_list); # then convert it as a string as parameter for c++ executable, this command insert space between each item on my computer
executable_command = executable_filename + data_str;# get the command

# now run the command
output = pexpect.run(executable_command);

# now parse the data
data_output = array([float(x) for x in output.split()]); # here you get your numpy data to play with




