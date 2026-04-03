# Polarity Classification

Run ```pip install -r requirements.txt``` to install the required dependencies.

Upload model files inside a folder named 'local_finetuned'. Get model files from [this link](https://entuedu-my.sharepoint.com/:f:/g/personal/ivansoji001_e_ntu_edu_sg/IgBRNMXQh2p7SJA0r7E8H3lxASxSlVDXRx2o7QFDxR6kn7M?e=9IDqX1).

## Overview

1. Input Processing: A single review or text string is passed into ```predict_single(text)```. The text is classified as a whole document. There is no sentence splitting or clause segmentation.
2. Model Prediction: Used RoBERTa-base transformer model fine-tuned on the Amazon Reviews dataset. The model is trained to classify reviews into two classes: positive and negative. The model outputs a predicted label for the input text.
3. SenticNet Prediction: SenticNet is a knowledge base that provides sentiment information for concepts. The model also uses SenticNet to extract sentiment information from the input text, which can help improve the accuracy of the classification. Computes average polarity score.
4. Weighted Voting: Each model contributes a vote:
- local_finetuned → weight = 4
- senticnet → weight = 1
- Only positive and negative predictions are counted.
5. Final Label: Determined by the weighted majority vote.


## Usage

```python

from polarity import PolarityClassifier

local_model_path = "./local_finetuned"

clf = PolarityClassifier(
    mode="local_finetuned_with_senticnet",
    local_model_path=local_model_path,
)

texts = [ # example reviews

    "i first bought a galaxy a54 the camera was an actual trash but with updates the camera good very good for what the phone can do however a 900 phone simply can't be the same the galaxy_s26 still produces worse dynamic range worse video worse color accuracy worse selfie and worse image quality when compared to the iphone samsung has an addiction to artificial intelligence video photo enhancements and they are currently squeezing the last molecule of what the galaxy_s22 could have do in order to increase their profit margins to the limits of the users this just doesn't happen with iphones iphones gives you practically what the sensor can do really good sensors and almost null ai enhancements are done to the images let's not speak about the selfie or zoom photos for context i've had a33 a54 a55 s24fe iphone_15 and now i only own one phone the iphone_15_pro the best camera phone i've ever had and i want to switch to the iphone_16_pro or iphone_17_pro soon as samsung released a pile of trash hardware from 4 years ago called galaxy_s26",

    "you should i write messy but imma give you a no filter review no in depth profound stuff just what i noticed raw after getting this phone from pre order early and coming from a mid range the switch was so underwhelming if i should list the changes i noticed the only changes i'd say is the smoothness the smooth animation compared to a mid range phone and the new ai capabilities kinda disappointed though by their eraser in editing as they removed the normal eraser and combined it with their ai eraser so removing a very little thing will give you a watermark that says the photo is ai generated or something regardless of how little you edited while the smoothness of it is noticeable it's not a change i'm so suprised and wow'ed by the camera too is soooo underwhelming the fact that i got it the same price as the flip 6 w discount and galaxy_s26 has a worst camera it's so meh disappointing whilst the flip has a way better camera quality the samsung mid range had an a55 it's so underwhelming it feels like it has the same camera quality as the galaxy_s26 base model i don't see much big difference if you're just going to look at it raw but if you tested it in detail there will be noticeable difference until with the flip the differentlce is huge and easy to notice so i'm not gagging that hard for this price difference? nope boring the only difference is the processing of details when you zoom in the galaxy_s26 is way better but honestly not life changing way better but like i said despite the difference it's not something i'm gagging over but the galaxy_s26 do got the vertical lock the only good change i noticed the base doesn't have the privacy screen feature obviously only the ultra got it in conclusion it's very underwhelming for me i actually thought of buying iphone_17 when i was waiting and contemplating before the galaxy_s26 release but as someone who's been mainly using samsung for almost a decade now i'm very used to its feature and there's no way that i'll be switching and paying for a phone like an iphone that lacks lots of things that makes using my phone easy for me i cannot pay that much and be more disappointed than being underwhelmed despite liking iphone's premium feels and stuff and i'd say for me the base iphone and pixel has a more noticable better camera quality than the galaxy_s26 i do not know their specs on paper but i prefer how clear pixel and iphone_17's camera is base on hands on exprience but would i still choose galaxy_s26 if i knew that before hand? probably yeah as someone who came from another samsung device i'm used to it and i feel in love with the light weight of it and small size like i said even if the other flag ship base models of apple and google are the same size because i want a smaller phone samsung has features for me that makes it easy and smarter to use puls the new and better galaxy ai capabilities i like it as someone who utilize with those assistance fetaures a lot not to disregard those new features but overall for me the switch is still underwhelming not totally disappointed not bad not at all but i'd say it's real just underwhelming but i feel like if you go for the ultra that will give a better noticeable change but if your older phone is still doing well and you're fine with that don't bother wait for a real upgrade i kinda hope they have a phone that physically feels like the galaxy_s26 base model the size the weight while having an s ultra features i would have gone for that",

    "dropping titanium from the newest galaxy_s26 line was actually a very good thing titanium is heavier than aluminum it also retains heat which is a problem you don't want in a phone using aluminum allows the phone to cool itself more efficiently titanium should never have been used on a phone samsung isn't the only one that dropped the titanium apple was the first to get rid of it last year and yes the galaxy_s26 base model is no longer considered a true flagship phone from samsung all phone manufacturers will push you to the largest and most advanced variant again apple does this with the iphone_17_pro and when it comes to the galaxy_s26_ultra specifically it matches or exceeds every galaxy foldable spec wise the only thing it doesn't do is fold which in my opinion is a very good thing i would rather have a slab phone that will last me a long time than a foldable that will break if it isn't handled with kid's gloves the base galaxy_s26 is the entry point into the s line samsung and other manufacturers will always hold back a feature or two to entice you to upgrade something apple is the king of when it comes to their computers for example when it comes to storage or ram almost any manufacturer will dangle a carrot in front of you to get you to upgrade"
    
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
i first bought a galaxy a54 the camera was an actual trash but with updates the camera good very good for what the phone can do however a 900 phone simply can't be the same the galaxy_s26 still produces worse dynamic range worse video worse color accuracy worse selfie and worse image quality when compared to the iphone samsung has an addiction to artificial intelligence video photo enhancements and they are currently squeezing the last molecule of what the galaxy_s22 could have do in order to increase their profit margins to the limits of the users this just doesn't happen with iphones iphones gives you practically what the sensor can do really good sensors and almost null ai enhancements are done to the images let's not speak about the selfie or zoom photos for context i've had a33 a54 a55 s24fe iphone_15 and now i only own one phone the iphone_15_pro the best camera phone i've ever had and i want to switch to the iphone_16_pro or iphone_17_pro soon as samsung released a pile of trash hardware from 4 years ago called galaxy_s26

Output:
negative

Input Text:
you should i write messy but imma give you a no filter review no in depth profound stuff just what i noticed raw after getting this phone from pre order early and coming from a mid range the switch was so underwhelming if i should list the changes i noticed the only changes i'd say is the smoothness the smooth animation compared to a mid range phone and the new ai capabilities kinda disappointed though by their eraser in editing as they removed the normal eraser and combined it with their ai eraser so removing a very little thing will give you a watermark that says the photo is ai generated or something regardless of how little you edited while the smoothness of it is noticeable it's not a change i'm so suprised and wow'ed by the camera too is soooo underwhelming the fact that i got it the same price as the flip 6 w discount and galaxy_s26 has a worst camera it's so meh disappointing whilst the flip has a way better camera quality the samsung mid range had an a55 it's so underwhelming it feels like it has the same camera quality as the galaxy_s26 base model i don't see much big difference if you're just going to look at it raw but if you tested it in detail there will be noticeable difference until with the flip the differentlce is huge and easy to notice so i'm not gagging that hard for this price difference? nope boring the only difference is the processing of details when you zoom in the galaxy_s26 is way better but honestly not life changing way better but like i said despite the difference it's not something i'm gagging over but the galaxy_s26 do got the vertical lock the only good change i noticed the base doesn't have the privacy screen feature obviously only the ultra got it in conclusion it's very underwhelming for me i actually thought of buying iphone_17 when i was waiting and contemplating before the galaxy_s26 release but as someone who's been mainly using samsung for almost a decade now i'm very used to its feature and there's no way that i'll be switching and paying for a phone like an iphone that lacks lots of things that makes using my phone easy for me i cannot pay that much and be more disappointed than being underwhelmed despite liking iphone's premium feels and stuff and i'd say for me the base iphone and pixel has a more noticable better camera quality than the galaxy_s26 i do not know their specs on paper but i prefer how clear pixel and iphone_17's camera is base on hands on exprience but would i still choose galaxy_s26 if i knew that before hand? probably yeah as someone who came from another samsung device i'm used to it and i feel in love with the light weight of it and small size like i said even if the other flag ship base models of apple and google are the same size because i want a smaller phone samsung has features for me that makes it easy and smarter to use puls the new and better galaxy ai capabilities i like it as someone who utilize with those assistance fetaures a lot not to disregard those new features but overall for me the switch is still underwhelming not totally disappointed not bad not at all but i'd say it's real just underwhelming but i feel like if you go for the ultra that will give a better noticeable change but if your older phone is still doing well and you're fine with that don't bother wait for a real upgrade i kinda hope they have a phone that physically feels like the galaxy_s26 base model the size the weight while having an s ultra features i would have gone for that

Output:
positive

Input Text:
dropping titanium from the newest galaxy_s26 line was actually a very good thing titanium is heavier than aluminum it also retains heat which is a problem you don't want in a phone using aluminum allows the phone to cool itself more efficiently titanium should never have been used on a phone samsung isn't the only one that dropped the titanium apple was the first to get rid of it last year and yes the galaxy_s26 base model is no longer considered a true flagship phone from samsung all phone manufacturers will push you to the largest and most advanced variant again apple does this with the iphone_17_pro and when it comes to the galaxy_s26_ultra specifically it matches or exceeds every galaxy foldable spec wise the only thing it doesn't do is fold which in my opinion is a very good thing i would rather have a slab phone that will last me a long time than a foldable that will break if it isn't handled with kid's gloves the base galaxy_s26 is the entry point into the s line samsung and other manufacturers will always hold back a feature or two to entice you to upgrade something apple is the king of when it comes to their computers for example when it comes to storage or ram almost any manufacturer will dangle a carrot in front of you to get you to upgrade

Output:
positive
```
