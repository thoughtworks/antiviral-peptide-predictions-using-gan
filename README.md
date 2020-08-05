# Antiviral peptide predictions using GAN

Recently a form of competitive deep neural networks have garnered a lot of attention. These GAN’s have a solution generation component and a competing network that tries to fool the models generation by the first network. This competition and evolution ensures both robust recognition of features that classify objects into given categories while also generating new variations of original input data. This methodology can be incorporated into a framework for generation of bioactive peptides. GAN’s work very well on images and strings. The challenge is to either build GAN’S for bioactive peptide generation from scratch using python based deep learning frameworks or customize existing GAN implementations developed for new molecule generation based on SMILES . Success criterion is a python pipeline that utilizes GAN’s to generate potential bioactive peptides < 2000 kDa. Framework will take bioactive antiviral peptides or similar as input and some randomized peptides as initial starting adversarial examples . Computational biologists and peptide chemistry experts will guide and verify results. An Additional non-mandatory extension, a simple GUI might be made available to run this pipeline and collect stored results at a later time. More advanced work can be accomplished in collaboration with structural biologists and computational chemists by using binding site, peptide characteristics and interaction by using advanced software alongside GAN’s

DDH link:
https://innovateindia.mygov.in/drug-ps/track-2-general-drug-discovery-including-covid/ddt2-10/

Dataset: https://www.hiv.lanl.gov/content/sequence/PEPTGEN/Explanation.html
Supporting Data guidelines: http://crdd.osdd.net/servers/avpdb/


