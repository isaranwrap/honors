
# Representing Fly Behavior with Recurrent Neural Networks 

### [Advisor](http://www.physics.emory.edu/home/people/bios/berman-gordon.html): Dr. Gordon Berman, Assistant Professor of Biology and Biomedical Engineering

## Introduction
My honors thesis work was on building a behavioral representation with recurrent neural networks (RNNs) in *D. Melanogaster*, the common fruit fly. 
All the code for training the networks I used, the preprocessing and resulting data analysis is contained in this notebook 
(although it's a bit messy - sorry I was hustling near the end what with corona & all x). I was in Dr. Gordon Berman's lab at Emory University,
in the Department of Biology (although I completed my honors in physics). An abridged version is included here - to read my full thesis see **honors_thesis[final].pdf** in the repository and of course, feel free to email isaranwrap@gmail.com for any questions! :) 

<p align=”center”>
  
![slide37](https://user-images.githubusercontent.com/30949122/90708986-06595400-e250-11ea-93e3-2da4f8970556.gif)
![slide43](https://user-images.githubusercontent.com/30949122/90708882-cabe8a00-e24f-11ea-92c9-230f56cf553d.gif)

</p>

## Abstract

The field of neuroscience has grown leaps and bounds in the past decade or so; 
quite a bit of work has been put into assembling and analyzing large swaths of neural data.
A good example of this is the field of connectomics - a connectome of *C. elegans* has been fully realized and one of *D. melanogaster* is soon on its way. However, a rich description of the anatomy and physiology of the brain does not immediately translate to an understanding of the output of the brain. If we consider the input the combination of the set of neuronal firing patterns and the environment an organism is placed in, the output would be something along the lines of behavior. The problem arises in the fact that there is no well-defined, precise, mathematically rich and meaningful description of behavior. Behavior operates on a variety of time and length scales, it is context-dependent and the set of external and internal stimuli which modulate behavior are not equally weighted - they are hierarchically ordered and dynamic. 

Recurrent neural networks (RNNs) offer a potentially viable solution to building a dynamical representation of the underlying forces driving the movement of *Drosophila melanogater*. The idea is that by building an RNN with many more parameters than necessary to capture the fly's movement, what will result is a chiefly low dimensional representation embedded in a high dimensional space. Then, the structure of the resulting manifold can be visualized with dimensionality reduction techniques to analyze clusters in the high dimensional space. One can dissect postural elements into `eigenflies` which can elicit low-dimensional representations early on in the analysis pipeline.  The hypothesis going into the honor's work was that the representation the RNN would build would be cleaner than the representation the raw postural time data yielded.

## More info

Perhaps the best place to get started if you are interested in learning more about the set of ethological techniques used starts with a [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028) published by Greg Stephens and colleagues in 2008, who apply postural analysis in _C. elegans._ The beauty of decomposing postural states into unit bases (plural) is the principal components associated with resulting eigenvectors describe `eigenworms`. The worms are moving in *sine* and *cosine* waves! Only once you see them moving in linear combinations of the sinuisoidal waves can you **_C_** the **_elegance_** in these nematodes. From there, understanding the different pipelines to flesh out the temporal correlations in behavior is part art, part science; a lot of fun!

## References
<a id = "1"> [1] </a>
Thomas Baumgartner et al. “The neural circuitry of a broken promise”.
In: Neuron 64.5 (2009), pp. 756–770.
