Images are pretty weird if you take your time to zoom in and look at them up close. They're just boxes with different brightness and yet we can easily recognize what they're representing in an instant. Have you ever wondered how many pixels do we need to recognize a feature in an image?

This post is a brief of the "BRIEF: Binary Robust Independent Elementary Features" paper written by Michael Calonder, Vincent Lepetit, Christoph Strecha, and Pascal Fua. Do read the paper if you are curious about more details. This post gets involved in increasing amount of Computer Science lingo towards the bottom so I guess uh suit yourself(?)

# Encoding Features
Take a look at the image below.

<img src="https://github.com/KamiliArsyad/posts/assets/22293969/959141bd-79ac-462c-ae6e-cdbc5f00bcc9" width="200">

Would you be able to construct an imagination of what this is? If you do, then well done! You're a pretty good feature recognizer and we no longer need to store the whole picture for you to store the information in the image. For those of you who don't know what this is, let's take a bit of a step back and see.

<img src="https://github.com/KamiliArsyad/posts/assets/22293969/55235f79-6c9b-4a09-b87f-0ab3b5afb8d2" width="200">

This looks like something metallic and, if the orientation is correct, it looks like it's some kind of a horizontal shiny cylinder. That's a lot of information to extract from a small image patch and that's great because it's probably better to store "horizontal metal cylinder" rather than the whole image.

In other words, we have essentially encoded this image patch as some words to describe the feature in that patch. If we ever find another patch of image with the same words/description associated with it as what we described this image with, you can lookup your database of words-to-image mapping quickly to perform feature matching and find another image with the same description.

Here are some more pixels for you, if you are still curious.

<img src="https://github.com/KamiliArsyad/posts/assets/22293969/12e02eb1-ccda-45d6-a246-a9cf9c3aa2d1" width="200">

This image gives you more words associated with it: Hand rail, staircase, emergency exit, metal rail, daylight, etc. However, notice that we're now talking about a set of features describing an image. This will come handy for another discussion later but I'm trying to discuss about elementary features so let's stick with a smaller patch. We see now that for most of our uses that deals with elementary features, instead of storing a picture, a small cut out of a picture, or a small patch of a picture, we can just store the encoded descriptor and perhaps the location of where we saw the corresponding feature to save memory.

## Need for Speed
The second image that I showed you just now consists of around 31x31 RGB pixels which was a lot to store. Each pixel consists of 8-bits of data for each color red, green, and blue and you can probably see that it's pretty useless to store everything as the color doesn't even seem to give you any useful information. Furthermore, this means we're storing 961 pixels × 3 colors/pixel × 8 bits/color = 23,064 bits just to say "shiny horizontal bar".

There is no exact definition of what a feature is but you have the idea what they are now and that we need a good way to store the features of an image in a compact format. We also need to somehow compare it to other features fast for matching algorithms. By fast here I mean fast fast. One normal video frame can easily give us 5000 features; for time-critical systems like SLAM, we are required to store and compare this 5000 features/frame × 30 frames/second = 150k features every second somehow. We also need to store them in an as small as possible format since storing the whole patch of each feature will fully occupy our computer's memory in a few seconds; might be better to just store a video at that point.

Another super painful problem in machine learning for computer vision (if we're going that way) is labelling. It's extremely tedious to label hundreds of thousands of features manually and thus we need to rely in an unsupervised feature encoding and comparison mechanism that does not require any manual intervention to match the features.

# BRIEF - Binary Robust Independent Elementary Features
BRIEF is an algorithm to produce binary descriptors. What I meant by binary descriptors is just a way to say that we encode a feature we want to encode in a binary string of some length. BRIEF algorithm is probably one of the most robust and most used algorithm nowadays to encode features in an image; the only catch is that it's so powerful that the way it works is kinda silly. Let me explain.

If we look at the following images of objects that looked kind of the same (e.g. they have close/similar features) in an image, can you think of a way to somehow encode the 'sameness' of these images (i.e. what makes them look similar)?
			
			
![image](https://github.com/KamiliArsyad/posts/assets/22293969/8968a626-8aa5-4cd0-a382-fc91a0ee9633)


Now I gotta stop you before you think too hard; let's delve a little into how the algorithm works and I'll tell you later why it works:
Assume that the image is grayscale, let p(c) where c is the coordinate of a pixel ϵ ℤ<sup>2</sup> denotes the intensity (brightness) of that pixel.
For a feature that we have identified (the feature locating algorithm is not explained here) and want to encode, we place a square patch of 31×31 pixels on that feature of the image.

On that square patch, pick two random (following a certain gaussian distribution) pixels p<sub>1</sub> and p<sub>2</sub> and mark them.

Bear with me here; define a test function τ: ℤ<sup>2</sup> → {0, 1} like so
τ(a, b) = 1 if p(a) < p(b), or 0 otherwise.

We then put our randomly selected pair of points p<sub>1</sub> and p<sub>2</sub> into τ like so:
τ(p<sub>1</sub>, p<sub>2</sub>)
and store the result in a bit.

Now we just need to keep doing this (pick random pair and compare) 256 times and we'll end up having 256 0s and 1s. We then simply store this as a 256-bit string and voila! We have our binary descriptor.

In total, we picked 256 random pairs of pixels ((p<sub>1,1</sub>, p<sub>2,1</sub>), (p<sub>1,2</sub>, p<sub>2,2</sub>), …, (p<sub>1,256</sub>, p<sub>2,256</sub>)) inside our square. This 256-pair tuple random pattern is stored and we shall use this same pattern over and over again for all the features that we want to encode.

## The Reasoning
I frankly don't know how the authors of the paper managed to come up with that binary feature idea; however, I might know how to make a sense out of it.

If we observe the 4 patches of a feature that I showed earlier, we can infer some sort of pattern which ultimately boils down to how the intensity of some regions of the patch compares to other regions of the same patch. In the case of the 4 patches, it is apparent that we need some sharp left-dark region with some areas sticking out in a zig-zag pattern. If I gave you the spoiler that we want binary features, then the idea of comparing pixels becomes more realistic.

The only problem with comparison, though, is we don't know which pair of pixels we should compare. We can get away with comparing each pixel with every other pixels in the patch with two interrelated issues that come up with this idea:
1. On a patch of size SxS, the resulting binary string of comparing each pixel against each other will obviously be of length Θ(S3). With S := 31, we'll need 29,791 bits which is worse than the 23K bits we needed earlier. This also comes at a cost of doing that much comparison and performing a lot of tedious memory management to support it. That's why we need less comparison than that.
2. However, if we decrease S to achieve less bits than what we calculated in the previous point, we lose spatial detail and there will be no feature. See for yourself:
	

Naturally we want less comparisons with the same amount of pixels as most of our brute-force comparison are useless anyway: for example, comparing a pixel to its first few immediate neighbors are most likely useless. We want the pixels compared to be kind of spaced apart from each other as much as we can, but we still want some of the pairs' ends to be close to each other just in case we need some close up details. You can start to see now why the Gaussian distribution thing was a brilliant but not an entirely random (pun not intended) idea. It is however not fun to jump into conclusions, so off the authors went to compare the performance of different patches:

After all the test whose most details were skipped in the paper for understandable conciseness reasons, they found out that distribution G II that is sampled from an isotropic Gaussian distribution with (X, Y) ~ i.i.d Gaussian(0, S2/25) gives the best result.

## Why We (Should) Love BRIEF
The patterns above consist of 128 tests and by using twice of that, 256-bits, BRIEF is practically perfect -maybe with a few caveats but let's talk about the goodness first.
1. BRIEF is easy to compute and parallelize. Sure, there are variations that involves triplets or quadruplets instead of pairs of pixels; but ultimately we're just making 256 comparisons over a fixed set of pairs that we had determined at the start. Feed this into a GPU with 256 CUDA cores and we can do all this comparison in an order of billionth of a second. With proper memory management, we can encode 50,000 features in an image within a millisecond on average.
2. BRIEF is lightweight. Storing a single feature only requires the same amount of storage as storing 4 doubles. This means that we can store tens of millions of features in a compact database that we can query in the order of microseconds to perform something that made BRIEF extremely powerful:
3. Comparison. To compare the difference between two features, we can do a classic hamming distance calculation on the bitstrings to see how many of the tests give different result. This dissimilarity measure naturally fulfil the mathematical properties of non-negativity, identity of indiscernibles, symmetry, and the triangle equality. Calculating hamming distance of two 256-bitstrings can be done in a few XOR operations and, depending on the instruction set of your processor, a few more clockcycles for summing the number of 1s in the result. These are all operations that can be easily parallelized to achieve thousands of comparisons in a microsecond.
The devil is of course in the detail of how to perform the required memory management, pipelining, and other optimizations to truly realize the full potential of this algorithm. Nevertheless, the exact performance upper bound can easily be estimated to be pretty good and not impossible to achieve.

# Big Cat, Small Cat, and Rotated Cat
Take a look at these three pictures

![image](https://github.com/KamiliArsyad/posts/assets/22293969/83a061a4-10ef-4b99-a5cc-e5575303fecd)
		
Can you tell the similarities between all of the pictures? First of all, cats are better than dogs. More importantly, all three pictures are feature a cat as their main object of interest. Last but not least, if you haven't noticed yet, they're all the same pictures. I do hope you noticed that because the point I'm trying to make here is that our brain, being the highly performance feature recognizer that they are, possess something called scale and rotational invariance. Regardless of how huge the picture is or how we rotate the picture, for the most part, we'll still be able to associate the same description to the picture. In our minds, a rotated cat is still a cat. Vanilla BRIEF, however, does not possess this ability.

For BRIEF, an inverted cat is a thing and a cat is another thing. Maybe I should stop saying cat since BRIEF operates on a 31x31 pixel patch and that can barely be a cat, but I am sure you understand why a feature and a rotated or scaled feature are different things for BRIEF. Take the same G(II) pattern on a patch and then use the pattern on the same but inverted patch, we will get a completely different binary descriptor. In fact, a few degrees of rotation is enough to completely mess up the descriptor and create a whole new descriptor with tremendous hamming distance from the unrotated version. Bummer.

Fret not, though, there is another improved version of BRIEF that has the nice rotational and scale invariance inspired by ponzi scheme. I might or might not make another post about that but I hope you learn a thing or two from this post. Oh also OpenCV has a BRIEF algorithm in `cv2::xfeatures2d::BriefDescriptorExtractor` in case you wanna try it out.

# Reference
	Calonder, M., Lepetit, V., Strecha, C., Fua, P. (2010). BRIEF: Binary Robust Independent Elementary Features. In: Daniilidis, K., Maragos, P., Paragios, N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6314. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-15561-1_56

