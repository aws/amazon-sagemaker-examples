03/12/2012

1. One-hundred plant species leaves data set.
	

2. Sources:
   (a) Original owners of colour Leaves Samples:

	James Cope, Thibaut Beghin, Paolo Remagnino, Sarah Barman.
	The colour images are not included in this submission.
	The Leaves were collected in the Royal Botanic Gardens, Kew, UK.
	email: james.cope@kingston.ac.uk
   
   (b) This dataset consists of work carried out by James Cope, Charles Mallah, and James Orwell.
	Donor of database Charles Mallah: charles.mallah@kingston.ac.uk; James Cope: 	james.cope@kingston.ac.uk

   (c) Date received 03/12/2012

3. Past Usage:

   (a) This is a new data set, provisional paper: 

	Charles Mallah, James Cope, James Orwell. Plant Leaf Classification Using Probabilistic Integration of Shape, Texture and Margin Features. Signal Processing, Pattern Recognition and Applications, in press.

   (b) Previous parts of the data set relate to feature extraction of leaves from: 

	J. Cope, P. Remagnino, S. Barman, and P. Wilkin.
	Plant texture classification using gabor cooccurrences.
	Advances in Visual Computing,
	pages 669–677, 2010.

	T. Beghin, J. Cope, P. Remagnino, and S. Barman.
	Shape and texture based plant leaf classification. In
	Advanced Concepts for Intelligent Vision Systems,
	pages 345–353. Springer, 2010.

4. Relevant Information Paragraph:
	The data directory contains the binary images (masks) of the leaf samples. The colour images are not included.
	The data set features are organised as the following:
	'data_Sha_64.txt'
	'data_Tex_64.txt'
	'data_Mar_64.txt'

	One file for each 64-element feature vectors. Each row begins with the class label.
	The remaining 64 elements is the feature vector.

5. Number of Instances

	1600 samples each of three features (16 samples per leaf class).

6. Number of Attributes

   	Three 64 element feature vectors per sample.

7. Vectors
	There are three features: Shape, Margin and Texture. As discussed in the paper(s) above.
	For Each feature, a 64 element vector is given per sample of leaf.
	These vectors are taken as a contigous descriptors (for shape) or histograms (for texture and margin).

8. Missing Attribute Values: none

9. Class Distribution: 16 instances per class