This framework is made to train and evaluate neural networks using a multitude of configurations, and select the optimal network configuration. 
Parallel training is achieved by submitting a high number of single core jobs on the T2 cluster. 

To run the framework, make an input python file, similar to the example "input.py" that is included. The input file should contain the name of the root file that includes a tree with signal events and a tree with background events. A list of variables to be used in the training should also be provided. Next to that, the input file contains several lists of network configurations that will be tried, these can be left at the default values. 

Once the input file is done simply run:

"python runTraining.py input_file_name"

Once all jobs are done, evaluate the ouptput by running:

"python OutputParser.py output_directory" 

from the directory where you launched the training jobs. Note that the output directory is automatically made depending on the name of your input file. This script will rank the top 10 neural networks, give their ROC integrals, and make a directory containing the weights of the top networks and plots showing their performance. 

If you want to train a parametric neural network, which is particularly useful for new physics searches, all you need is a branch in the signal root tree that has the new physics parameter(s) of interest. Say you have a root file "root_file", with a tree "signal_tree" that has the branch "m_newPhysics". To randomly sample this parameter from your signal distribution and add it to the background tree, contained for example in "background_tree", you simply run:

"python ParematerAddition.py root_file_name signal_tree background_tree m_newPhysics"

Once this is done, you can add "m_newPhysics" to the list of input variables of the input python file, after which you will be training parametric neural networks.
