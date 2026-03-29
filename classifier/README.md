# Polarity Classification

Run ```pip install -r requirements.txt``` to install the required dependencies.

Upload model files inside a folder named 'local_finetuned'. Get model files from [this link](https://entuedu-my.sharepoint.com/:f:/g/personal/ivansoji001_e_ntu_edu_sg/IgBRNMXQh2p7SJA0r7E8H3lxASxSlVDXRx2o7QFDxR6kn7M?e=E0gsfA).

## Usage

```python

from polarity import PolarityClassifier

local_model_path = "./local_finetuned"

clf = PolarityClassifier(
    mode="local_finetuned_with_senticnet",
    local_model_path=local_model_path,
)

texts = [ # example reviews

    "@felicioimWhen I first bought a Galaxy A54 the camera was an actual trash, but with updates the camera good very good for what the phone can do, however, a $ 900 phone simply can't be the same. The S26 still produces worse dynamic range, worse video, worse color accuracy, worse selfie, and worse image quality when compared to the iPhone. Samsung has an addiction to Artificial Intelligence video/photo enhancements, and they are currently squeezing the last molecule of what the S22 could have do in order to increase their profit margins to the limits of the users. This just doesn't happen with iPhones. iPhones gives you practically what the sensor can do, really good sensors, and almost null AI Enhancements are done to the images. Let's not speak about the selfie or zoom photos. For context, I've had: A33, A54, A55, S24FE, iPhone 15, and now I only own one phone: The iPhone 15 Pro. The best camera/phone i've ever had, and I want to switch to the iPhone 16 Pro or 17 Pro soon, as Samsung released a pile of trash hardware from 4 years ago called S26.",

    "Strong disagree. It is easy to compare S25 vs S26, or even S22 vs S26 and say 'Samsung is slacking', but please, let's blame Samsung after analyzing the competition once. iPhone 17 - 120Hz! Finally! YAY! But still uses USB 2.0, still charges at 25 Watts, still has a dual-camera setup. Hardware-wise, that is still NOT par with the Galaxy S20 from 2020, which had 120Hz, USB 3.0, 45-Watt charging and a 3-camera setup with 8K ability! Pixel 10 is far more competitive - 120Hz, USB 3.0, faster 30W charging, and the brand new 5x Optical Zoom lens. This does bring some competition to Samsung, but the Tenor G5 is still somewhat behind the Snapdragon 8 Gen 2 from Galaxy S23 series! So even this allows Samsung to still be a strong contender without doing any real work. So yeah, complain all you want about the S26 being 'boring' but Samsung has no reason to make any improvements when the competition is struggling to out-do what they did in 2020. Also do ask yourself - Why was the world singing praises for the iPhone 17, if the Samsung is underwhelming according to you? Just the Apple effect?",

    "Dropping titanium from the newest Galaxy S26 line was actually a very good thing. Titanium is heavier than aluminum. It also retains heat, which is a problem you don't want in a phone. Using aluminum allows the phone to cool itself more efficiently. Titanium should never have been used on a phone. Samsung isn't the only one that dropped the titanium. Apple was the first to get rid of it last year. And yes, the S26 base model is no longer considered a true flagship phone from Samsung. All phone manufacturers will push you to the largest and most advanced variant. Again, Apple does this with the iPhone 17 Pro Max. And when it comes to the Galaxy S26 Ultra specifically, it matches or exceeds every Galaxy foldable spec wise. The only thing it doesn't do is fold. Which, in my opinion, is a very good thing. I would rather have a slab phone that will last me a long time than a foldable that will break if it isn't handled with kid's gloves. The base S26 is the entry point into the S line. Samsung and other manufacturers will always hold back a feature or two to entice you to upgrade. Something Apple is the king of when it comes to their computers for example when it comes to storage or RAM. Almost any manufacturer will dangle a carrot in front of you to get you to upgrade."
    
]

for text in texts:
    print("\nInput Text:")
    print(text)
    print("\nOutput:")
    print(clf.predict_single(text)['label'])
```

## Output(s)

```
Input Text:
@felicioimWhen I first bought a Galaxy A54 the camera was an actual trash, but with updates the camera good very good for what the phone can do, however, a $ 900 phone simply can't be the same. The S26 still produces worse dynamic range, worse video, worse color accuracy, worse selfie, and worse image quality when compared to the iPhone. Samsung has an addiction to Artificial Intelligence video/photo enhancements, and they are currently squeezing the last molecule of what the S22 could have do in order to increase their profit margins to the limits of the users. This just doesn't happen with iPhones. iPhones gives you practically what the sensor can do, really good sensors, and almost null AI Enhancements are done to the images. Let's not speak about the selfie or zoom photos. For context, I've had: A33, A54, A55, S24FE, iPhone 15, and now I only own one phone: The iPhone 15 Pro. The best camera/phone i've ever had, and I want to switch to the iPhone 16 Pro or 17 Pro soon, as Samsung released a pile of trash hardware from 4 years ago called S26.

Output:
negative

Input Text:
Strong disagree. It is easy to compare S25 vs S26, or even S22 vs S26 and say 'Samsung is slacking', but please, let's blame Samsung after analyzing the competition once. iPhone 17 - 120Hz! Finally! YAY! But still uses USB 2.0, still charges at 25 Watts, still has a dual-camera setup. Hardware-wise, that is still NOT par with the Galaxy S20 from 2020, which had 120Hz, USB 3.0, 45-Watt charging and a 3-camera setup with 8K ability! Pixel 10 is far more competitive - 120Hz, USB 3.0, faster 30W charging, and the brand new 5x Optical Zoom lens. This does bring some competition to Samsung, but the Tenor G5 is still somewhat behind the Snapdragon 8 Gen 2 from Galaxy S23 series! So even this allows Samsung to still be a strong contender without doing any real work. So yeah, complain all you want about the S26 being 'boring' but Samsung has no reason to make any improvements when the competition is struggling to out-do what they did in 2020. Also do ask yourself - Why was the world singing praises for the iPhone 17, if the Samsung is underwhelming according to you? Just the Apple effect?

Output:
positive

Input Text:
Dropping titanium from the newest Galaxy S26 line was actually a very good thing. Titanium is heavier than aluminum. It also retains heat, which is a problem you don't want in a phone. Using aluminum allows the phone to cool itself more efficiently. Titanium should never have been used on a phone. Samsung isn't the only one that dropped the titanium. Apple was the first to get rid of it last year. And yes, the S26 base model is no longer considered a true flagship phone from Samsung. All phone manufacturers will push you to the largest and most advanced variant. Again, Apple does this with the iPhone 17 Pro Max. And when it comes to the Galaxy S26 Ultra specifically, it matches or exceeds every Galaxy foldable spec wise. The only thing it doesn't do is fold. Which, in my opinion, is a very good thing. I would rather have a slab phone that will last me a long time than a foldable that will break if it isn't handled with kid's gloves. The base S26 is the entry point into the S line. Samsung and other manufacturers will always hold back a feature or two to entice you to upgrade. Something Apple is the king of when it comes to their computers for example when it comes to storage or RAM. Almost any manufacturer will dangle a carrot in front of you to get you to upgrade.

Output:
positive
```
