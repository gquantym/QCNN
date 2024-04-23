import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def optimized_probabilities(opt_prob_non_pulsar,opt_prob_pulsar,epoch,title,num_sets):
    #Plotting time
    linspace_non_pulsar = np.linspace(0, len(opt_prob_non_pulsar), len(opt_prob_non_pulsar),dtype = int)/len(opt_prob_non_pulsar) #Linspace of samplesize 
    plt.scatter(linspace_non_pulsar, opt_prob_non_pulsar, c='blue', label = "Non-Pulsars") #Plots non-pulsars
    linspace_pulsar = np.linspace(0, len(opt_prob_pulsar), len(opt_prob_pulsar),dtype = int)/len(opt_prob_pulsar) #Linspace of samplesize 
    plt.scatter(linspace_pulsar, opt_prob_pulsar, c='red', label = "Pulsars")  #Plots pulsars
    
    plt.title("{0} - Optimized Probabilities - Epoch {1} - Set Number {2}".format(title,epoch,num_sets))
    plt.xlabel("Normalized samples")
    plt.ylabel("Optimized Probability")
    #plt.axhline(y=threshold_spec, color='g', linestyle='--', label='Mean specificity threshold = {:.1f}'.format(threshold_spec))
    #plt.axhline(y=threshold_acc, color='orange', linestyle='--', label='Mean accuracy threshold = {:.1f}'.format(threshold_acc))
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.4, -0.14))
    #plt.savefig('{0} - opt_probability - num set {1}'.format(title,num_sets),dpi=300,bbox_inches='tight')
    #plt.show()

def loss_function(epoch,loss_array,title,num_sets):
    linspace_epoch = np.linspace(0, epoch, epoch,dtype = int) #Linspace of samplesize 
    plt.plot(linspace_epoch, loss_array, c='green')  #Plots pulsars
    plt.title("{0} - Repetition Loss against Epoch {1} - Set Number {2}".format(title,epoch,num_sets))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1, 0.7))
    #plt.savefig('{0} - loss epoch {1} - set number {2}'.format(title,epoch,num_sets),dpi=300,bbox_inches='tight')
    #plt.show()

def test_probabilities(pulsar_probability,non_pulsar_probability,epoch,title,num_sets):
    linspace_pulsar = np.linspace(0, len(pulsar_probability), len(pulsar_probability),dtype = int)/len(pulsar_probability) #Linspace of samplesize 
    linspace_non_pulsar = np.linspace(0, len(non_pulsar_probability), len(non_pulsar_probability),dtype = int)/len(non_pulsar_probability)
    plt.scatter(linspace_non_pulsar, non_pulsar_probability, c='blue', label = "Non-Pulsars") #Plots non-pulsars
    plt.scatter(linspace_pulsar, pulsar_probability, c='red', label = "Pulsars")  #Plots pulsars
    #plt.axhline(y=spec_threshold, color='g', linestyle='--', label='Specificity threshold = {:.1f}\n'.format(spec_threshold))
    #plt.axhline(y=acc_threshold, color='orange', linestyle='--', label='Accuracy threshold = {:.2f}'.format(acc_threshold))
    #plt.axhline(y=spec_threshold, color='green', linestyle='--', label='Specificity threshold = {:.2f}'.format(spec_threshold))
    plt.title("{0} - Test Data - Epoch {1} - Set Number {2}".format(title,epoch,num_sets))
    plt.xlabel("Normalized Samples")
    plt.ylabel("Probability")
    #plt.legend(bbox_to_anchor=(1, 0.5))
    plt.ylim(0, 1)
    #plt.savefig('{0} - test_probability - num set {1}'.format(title,num_sets),dpi=300,bbox_inches='tight')
    #plt.show()
    
def plot_all(train_pulsar_prob,train_non_pulsar_prob,test_pulsar_prob,test_non_pulsar_prob,epoch, results, loss_array, title, num_sets,fpr, tpr):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Assuming similar parameters for these functions, adjust if different
    plt.sca(axs[0, 0])
    optimized_probabilities(train_non_pulsar_prob,train_pulsar_prob,epoch,title,num_sets)  # Adjust parameters as needed
    axs[0, 0].set_title("Trained Probabilities")

    plt.sca(axs[0, 1])
    loss_function(epoch, loss_array, title, num_sets)
    axs[0, 1].set_title("Loss Function")

    plt.sca(axs[1, 1])
    test_probabilities(test_pulsar_prob,test_non_pulsar_prob,epoch,title,num_sets)  # Adjust parameters as needed
    axs[1, 1].set_title("Test Probabilities")

    '''
    # Use the bottom left subplot for text
    axs[1, 0].axis('off')  # Ensure no axis is visible

    # Format and display results in the text subplot
    text_content = '\n'.join([f"{key.capitalize()}: {value:.4f}" for key, value in results.items()])
    axs[1, 0].text(0.5, 0.5, text_content, transform=axs[1, 0].transAxes,
                   ha='center', va='center', fontsize=10, color='black', family='monospace')
    '''
    '''
    roc_auc = auc(fpr, tpr)
    plt.sca(axs[1, 0])
    roc(fpr, tpr, roc_auc, num_sets, title)  # Adjust parameters as needed
    axs[1, 0].set_title("ROC")
    '''
    plt.sca(axs[1, 0])
    roc_auc = auc(fpr, tpr)
    roc(axs[1, 0], fpr, tpr, roc_auc, num_sets, title)
    axs[1, 0].set_title("ROC")
    
    # Add an overall title
    fig.suptitle(title, fontsize=16)
    plt.savefig('{0} - {1}'.format(title,num_sets),dpi=300,bbox_inches='tight')
    plt.show()


def probabilities(probability_pulsar,probability_non_pulsar,title,num_sets):
    linspace_non_pulsar = np.linspace(0, len(probability_non_pulsar), len(probability_non_pulsar),dtype = int)/len(probability_non_pulsar) #Linspace of samplesize 
    linspace_pulsar = np.linspace(0, len(probability_pulsar), len(probability_pulsar),dtype = int)/len(probability_pulsar) #Linspace of samplesize 
    plt.scatter(linspace_non_pulsar, probability_non_pulsar, c='blue', label = "Non-Pulsars") #Plots non-pulsars
    plt.scatter(linspace_pulsar, probability_pulsar, c='red', label = "Pulsars")  #Plots pulsars
    plt.title("{0} - Before Optimization - Set Number {1}".format(title,num_sets))
    plt.xlabel("Normalized Samples")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(1, 0.5))
    #plt.ylim(0, 1)
    #plt.savefig('{0} - Probability - Set Number {1}'.format(title,num_sets),dpi=300,bbox_inches='tight')
    plt.show()
def roc(ax, fpr, tpr, roc_auc, set_number, title):
    # Plot ROC curve directly on the provided axes object
    lw = 2
    ax.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic #{set_number}')
    ax.legend(loc="lower right")

'''
def roc(fpr,tpr,roc_auc,set_number,title):
    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic #{}'.format(set_number))
    plt.legend(loc="lower right")
    #plt.savefig('ROC {0} '.format(title,set_number),dpi=300,bbox_inches='tight')
    #plt.show()
    
'''
    
def test_prob_with_mretics(pulsar_probability,non_pulsar_probability,epoch,thresholds,title,num_sets):
    metric_names = ["Specificity","Accuracy","Recall","Precision","NPV","Balanced accuracy","Geometric mean","Informedness"]
    colours = ["green","blue","red","orange","magenta","purple","cyan","brown"]
    linspace_pulsar = np.linspace(0, len(pulsar_probability), len(pulsar_probability),dtype = int)/len(pulsar_probability) #Linspace of samplesize 
    linspace_non_pulsar = np.linspace(0, len(non_pulsar_probability), len(non_pulsar_probability),dtype = int)/len(non_pulsar_probability)
    plt.scatter(linspace_non_pulsar, non_pulsar_probability, c='blue', label = "Non-Pulsars") #Plots non-pulsars
    plt.scatter(linspace_pulsar, pulsar_probability, c='red', label = "Pulsars")  #Plots pulsars
    #plt.axhline(y=spec_threshold, color='g', linestyle='--', label='Specificity threshold = {:.1f}\n'.format(spec_threshold))
    for i,threshold in enumerate(thresholds):
        plt.axhline(y=threshold, color=colours[i], linestyle='--', label='{} threshold = {:.2f}'.format(metric_names[i],threshold))
    #plt.axhline(y=spec_threshold, color='green', linestyle='--', label='Specificity threshold = {:.2f}'.format(spec_threshold))
    plt.title("{0} - Test Data - Epoch {1} - Set Number {2}".format(title,epoch,num_sets))
    plt.xlabel("Normalized Samples")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(1, 0.9))
    plt.ylim(0, 1)
    #plt.savefig('{0} - Test Data w/ Optimized Weigths - num set {1}'.format(title,num_sets),dpi=300,bbox_inches='tight')
    plt.show()