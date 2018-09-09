# cycleGAN_denoising
Low Dose CT Image Denoising Using a Cycle-Consistent Adversarial Networks.

<img src="https://github.com/SSinyu/cycleGAN_denoising/blob/master/img/model.PNG"> 

- X : LDCT (64x64 patch extracted from a 512x512 image.)
- Y : NDCT (64x64 patch extracted from a 512x512 image.)

* Dataset : https://www.aapm.org/GrandChallenge/LowDoseCT/
--------------

### Training Loss (~100 epoch)
<img src="https://github.com/SSinyu/cycleGAN_denoising/blob/master/img/cycle_gan_loss.png">  

--------------

### Result
<img src="https://github.com/SSinyu/cycleGAN_denoising/blob/master/img/measure.png">

----|**LDCT**|20ep|40ep|60ep|80ep|100ep|
----|----|----|----|----|----|----
PSNR|**45.1087(1.5780)**|33.6883(2.3005)|34.6436(2.4673)|**38.0709(0.7668)**|37.0799(1.1700)|36.6317(1.2361)|
SSIM|**0.9708(0.0107)**|0.7542(0.1212)|0.8087(0.1116)|**0.9428(0.0112)**|0.8889(0.0305)|0.8772(0.0267)|
RMSE|**23.1228(4.1781)**|87.8086(24.3624)|78.8303(20.7045)|**51.3466(4.5541)**|57.8417(7.6144)|60.9726(8.5801)|



