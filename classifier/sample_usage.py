import os
import warnings

from transformers import logging

from polarity import PolarityClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def run_local_finetuned_example():
    print("\n=== LOCAL FINE-TUNED MODEL EXAMPLE ===")

    # Change this to your actual saved model folder
    local_model_path = "./local_finetuned"

    if not os.path.isdir(local_model_path):
        print(f"Local model folder not found: {local_model_path}")
        print("Update local_model_path to your saved fine-tuned model directory.")
        return
    '''
    clf = PolarityClassifier(
        mode="local_finetuned",
        local_model_path=local_model_path,
    )
    '''

    clf = PolarityClassifier(
        mode="local_finetuned_with_senticnet",
        local_model_path=local_model_path,
    )
    '''
    texts = [
        "I love the camera on this phone.",
        "This phone is terrible and slow.",
        "The phone was released last year.",
    ]
    '''
    '''
    texts = [
        "@felicioimWhen I first bought a Galaxy A54 the camera was an actual trash, but with updates the camera good very good for what the phone can do, however, a $ 900 phone simply can't be the same. The S26 still produces worse dynamic range, worse video, worse color accuracy, worse selfie, and worse image quality when compared to the iPhone. Samsung has an addiction to Artificial Intelligence video/photo enhancements, and they are currently squeezing the last molecule of what the S22 could have do in order to increase their profit margins to the limits of the users. This just doesn't happen with iPhones. iPhones gives you practically what the sensor can do, really good sensors, and almost null AI Enhancements are done to the images. Let's not speak about the selfie or zoom photos. For context, I've had: A33, A54, A55, S24FE, iPhone 15, and now I only own one phone: The iPhone 15 Pro. The best camera/phone i've ever had, and I want to switch to the iPhone 16 Pro or 17 Pro soon, as Samsung released a pile of trash hardware from 4 years ago called S26.",
        "Strong disagree. It is easy to compare S25 vs S26, or even S22 vs S26 and say 'Samsung is slacking', but please, let's blame Samsung after analyzing the competition once. iPhone 17 - 120Hz! Finally! YAY! But still uses USB 2.0, still charges at 25 Watts, still has a dual-camera setup. Hardware-wise, that is still NOT par with the Galaxy S20 from 2020, which had 120Hz, USB 3.0, 45-Watt charging and a 3-camera setup with 8K ability! Pixel 10 is far more competitive - 120Hz, USB 3.0, faster 30W charging, and the brand new 5x Optical Zoom lens. This does bring some competition to Samsung, but the Tenor G5 is still somewhat behind the Snapdragon 8 Gen 2 from Galaxy S23 series! So even this allows Samsung to still be a strong contender without doing any real work. So yeah, complain all you want about the S26 being 'boring' but Samsung has no reason to make any improvements when the competition is struggling to out-do what they did in 2020. Also do ask yourself - Why was the world singing praises for the iPhone 17, if the Samsung is underwhelming according to you? Just the Apple effect?",
        "Dropping titanium from the newest Galaxy S26 line was actually a very good thing. Titanium is heavier than aluminum. It also retains heat, which is a problem you don't want in a phone. Using aluminum allows the phone to cool itself more efficiently. Titanium should never have been used on a phone. Samsung isn't the only one that dropped the titanium. Apple was the first to get rid of it last year. And yes, the S26 base model is no longer considered a true flagship phone from Samsung. All phone manufacturers will push you to the largest and most advanced variant. Again, Apple does this with the iPhone 17 Pro Max. And when it comes to the Galaxy S26 Ultra specifically, it matches or exceeds every Galaxy foldable spec wise. The only thing it doesn't do is fold. Which, in my opinion, is a very good thing. I would rather have a slab phone that will last me a long time than a foldable that will break if it isn't handled with kid's gloves. The base S26 is the entry point into the S line. Samsung and other manufacturers will always hold back a feature or two to entice you to upgrade. Something Apple is the king of when it comes to their computers for example when it comes to storage or RAM. Almost any manufacturer will dangle a carrot in front of you to get you to upgrade."
    ]
    '''
    '''
    texts = [
        "i first bought a galaxy a54 the camera was an actual trash but with updates the camera good very good for what the phone can do however a 900 phone simply can't be the same the galaxy_s26 still produces worse dynamic range worse video worse color accuracy worse selfie and worse image quality when compared to the iphone samsung has an addiction to artificial intelligence video photo enhancements and they are currently squeezing the last molecule of what the galaxy_s22 could have do in order to increase their profit margins to the limits of the users this just doesn't happen with iphones iphones gives you practically what the sensor can do really good sensors and almost null ai enhancements are done to the images let's not speak about the selfie or zoom photos for context i've had a33 a54 a55 s24fe iphone_15 and now i only own one phone the iphone_15_pro the best camera phone i've ever had and i want to switch to the iphone_16_pro or iphone_17_pro soon as samsung released a pile of trash hardware from 4 years ago called galaxy_s26",
        "you should i write messy but imma give you a no filter review no in depth profound stuff just what i noticed raw after getting this phone from pre order early and coming from a mid range the switch was so underwhelming if i should list the changes i noticed the only changes i'd say is the smoothness the smooth animation compared to a mid range phone and the new ai capabilities kinda disappointed though by their eraser in editing as they removed the normal eraser and combined it with their ai eraser so removing a very little thing will give you a watermark that says the photo is ai generated or something regardless of how little you edited while the smoothness of it is noticeable it's not a change i'm so suprised and wow'ed by the camera too is soooo underwhelming the fact that i got it the same price as the flip 6 w discount and galaxy_s26 has a worst camera it's so meh disappointing whilst the flip has a way better camera quality the samsung mid range had an a55 it's so underwhelming it feels like it has the same camera quality as the galaxy_s26 base model i don't see much big difference if you're just going to look at it raw but if you tested it in detail there will be noticeable difference until with the flip the differentlce is huge and easy to notice so i'm not gagging that hard for this price difference? nope boring the only difference is the processing of details when you zoom in the galaxy_s26 is way better but honestly not life changing way better but like i said despite the difference it's not something i'm gagging over but the galaxy_s26 do got the vertical lock the only good change i noticed the base doesn't have the privacy screen feature obviously only the ultra got it in conclusion it's very underwhelming for me i actually thought of buying iphone_17 when i was waiting and contemplating before the galaxy_s26 release but as someone who's been mainly using samsung for almost a decade now i'm very used to its feature and there's no way that i'll be switching and paying for a phone like an iphone that lacks lots of things that makes using my phone easy for me i cannot pay that much and be more disappointed than being underwhelmed despite liking iphone's premium feels and stuff and i'd say for me the base iphone and pixel has a more noticable better camera quality than the galaxy_s26 i do not know their specs on paper but i prefer how clear pixel and iphone_17's camera is base on hands on exprience but would i still choose galaxy_s26 if i knew that before hand? probably yeah as someone who came from another samsung device i'm used to it and i feel in love with the light weight of it and small size like i said even if the other flag ship base models of apple and google are the same size because i want a smaller phone samsung has features for me that makes it easy and smarter to use puls the new and better galaxy ai capabilities i like it as someone who utilize with those assistance fetaures a lot not to disregard those new features but overall for me the switch is still underwhelming not totally disappointed not bad not at all but i'd say it's real just underwhelming but i feel like if you go for the ultra that will give a better noticeable change but if your older phone is still doing well and you're fine with that don't bother wait for a real upgrade i kinda hope they have a phone that physically feels like the galaxy_s26 base model the size the weight while having an s ultra features i would have gone for that",
        "dropping titanium from the newest galaxy_s26 line was actually a very good thing titanium is heavier than aluminum it also retains heat which is a problem you don't want in a phone using aluminum allows the phone to cool itself more efficiently titanium should never have been used on a phone samsung isn't the only one that dropped the titanium apple was the first to get rid of it last year and yes the galaxy_s26 base model is no longer considered a true flagship phone from samsung all phone manufacturers will push you to the largest and most advanced variant again apple does this with the iphone_17_pro and when it comes to the galaxy_s26_ultra specifically it matches or exceeds every galaxy foldable spec wise the only thing it doesn't do is fold which in my opinion is a very good thing i would rather have a slab phone that will last me a long time than a foldable that will break if it isn't handled with kid's gloves the base galaxy_s26 is the entry point into the s line samsung and other manufacturers will always hold back a feature or two to entice you to upgrade something apple is the king of when it comes to their computers for example when it comes to storage or ram almost any manufacturer will dangle a carrot in front of you to get you to upgrade"
    ]
    '''
    texts = [
        "strong disagree it is easy to compare galaxy_s25 vs galaxy_s26 or even galaxy_s22 vs galaxy_s26 and say samsung is slacking but please let's blame samsung after analyzing the competition once iphone_17 120hz! finally! yay! but still uses usb 2__ __0 still charges at 25 watts still has a dual camera setup hardware wise that is still not par with the galaxy_s20 from 2020 which had 120hz usb 3__ __0 45 watt charging and a 3 camera setup with 8k ability! pixel_10 is far more competitive 120hz usb 3__ __0 faster 30w charging and the brand new 5x optical zoom lens this does bring some competition to samsung but the tenor g5 is still somewhat behind the snapdragon 8 gen 2 from galaxy_s23 series! so even this allows samsung to still be a strong contender without doing any real work so yeah complain all you want about the galaxy_s26 being boring but samsung has no reason to make any improvements when the competition is struggling to out do what they did in 2020 also do ask yourself why was the world singing praises for the iphone_17 if the samsung is underwhelming according to you? just the apple effect?",
        "to be honest i feel like they are going in a sort of app direction where apple focused more on optimizing thier battery rather than increasing it which is why its always neck and neck with samsung even though has a smaller battery it still goes with it wil life span if you want an upgrade go for another android phone at this point i herd thr transfer between android s is pretty good so theres no stoping you from an upgrade edit i was also going to mention its cheap and goes to a everyday use phone kind of branch rather than the showoff kind of branch that alot of phone brands a going like i phone huawei samsung and a couple more which is why i just feel like the galaxy_s26 is not really appealing and is missing alot of stuff and feels more like a phone that tries to tempt people in the upgrade into it",
        "the 1000 price for the galaxy_s26 with only 25w charging in the eu is honestly hard to justify i've been a long time samsung user i had the s10e please make smaller phones great again then the galaxy_s22 which i replaced very quickly because the battery life and the exynos chip were so disappointing right now i'm using the galaxy_s23 and i've had it for almost three years it still works just as well as it did on day one which shows how solid the device is however looking at the galaxy_s26 i simply don't see any real reason to upgrade as someone who has been a fan of samsung phones for years it's frustrating to feel like the company is no longer listening to its customers at this point it honestly feels like samsung has become even worse than apple in some aspects"
    ]
    

    for text in texts:
        print("\nInput Text:")
        print(text)
        print("\nOutput:")
        #print(clf.predict_single(text))
        print(clf.predict_single(text)['label'])


if __name__ == "__main__":
    run_local_finetuned_example()