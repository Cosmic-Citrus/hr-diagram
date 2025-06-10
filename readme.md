# Repo:    hr-diagram

The purpose of this code is to generate Hertzsprung-Russell (HR) Diagrams.

## Description

The HR Diagram can be used to model stellar evolution. The location of a star on the HR Diagram will change as the star enters different stages of its life-cycle. Repeating this procedure for multiple stars reveals some trends that are useful for classifying them into different groups.

<img src="file:///Users/owner/Desktop/programming/hr_diagram/output/example_01-data_via_web_scrape/HR_Diagram-LuminosityClass-AbsoluteMagnitude_VS_ColorIndex(BV)-wSun-wIsoRadius-wSpectralTypes_MarkAt9V.png" title="" alt="example-hr-diagram-basic" data-align="center">

An explanation of relevant parameters is below:

* Surface temperature $T$
  
  * measures the average temperature at the surface of a star, typically in Kelvin $K$
  
  * for reference, the surface temperature of the Sun is roughly $5800 K$

* Color Index $B-V$
  
  * measures the difference between successive measurements of the visual magnitude through B and V filters, where B is sensitive to blue light and V is sensitive to visible green-yellow light
  
  * hotter stars appear bluer and cooler stars appear redder
  
  * assuming that the star is an ideal blackbody, $T$ and $B-V$ are related by
    
    $T = 4,600 \, \text{K} \left[ \frac{1}{0.92(B-V) + 1.7} + \frac{1}{0.92(B-V) + 0.62} \right]$

* Absolute magnitude $M$
  
  * measures the intrinsic brightness of a star; this is defined to be the apparent brightness of a star as viewed from Earth that is 10 parsecs (roughly 3.26 light years) away
  
  * uses an inverse logarithmic scale
    
    * For example, $M_{Sun} = 4.83$ and $M_{Regulus A} = -0.57$ ⇒ Regulus A is intrinsically brighter than the Sun

* Surface Area $A_{S}$
  
  * measures the area of the surface of a star
  
  * for a sphere of radius R, the surface area is
    
    $A_{S} = 4 \pi R^{2}$

* Luminosity $L$
  
  * measures the total power output (energy per unit-time) of a star, typically in units relative to the average luminosity of the Sun $L_{\odot} = 3.828 \times 10^{26}$ $W$
  
  * increases as $T$ increases; decreases as $T$ decreases
  
  * increases as $A_{S}$ increases; decreases as $A_{S}$ decreases
  
  * is related to $M$ by 
    
    $M = -2.5 \log\left(\frac{L}{L_{\odot}}\right) + M_{\odot}$
  
  * assuming that the star is an ideal blackbody, $L$ is related to $T$ and $A_{S}$ by the Stefan-Boltzmann Law, given by
    
    $L = A_{S} \sigma T^{4}$
    
    where 
    
    Stefan-Boltzmann constant $\sigma = \frac{2 \pi^{5} k_{B}^{4}}{15 c^{2} h^{3}} \approx 5.670 \times 10^{-8} \frac{W}{m^{2} K^{4}}$ 
    
    Boltzmann constant $k_{B} \approx 1.38 \frac{J}{K}$
    
    speed of light $c \approx 3 \times 10^8 \frac{m}{s}$
    
    Planck constant $h \approx 6.626 \times 10^{-34} \frac{J}{Hz}$

* Spectral classification and sub-classification
  
  * stars can be separated into stellar classes and sub-classes, which correspond to $T$ and other parameters (such as metallicity, etc)
  
  * most stellar objects can be fit into $O$, $B$, $A$, $F$, $G$, $K$, and $M$ stellar classifications, where $O$ stars are much hotter than $M$ stars
  
  * each stellar classification is sub-divided into 10 partitions ($0 - 9$), where $0$ is hotter than $9$ 

* Luminosity classes
  
  * account for spectral features, such as the width of absorption lines, which relate to the surface gravity of the star
    
    * $0$ or $Ia+$
      
      * hypergiants and extremely luminous giants
    
    * $Ia$
      
      * luminous super-giants
    
    * $Iab$
      
      * intermediate-sized luminous super-giants
    
    * $Ib$
      
      * less luminous super-giants
    
    * $II$
      
      * bright giants
    
    * $III$
      
      * normal giants
    
    * $IV$
      
      * sub-giants
    
    * $V$
      
      * main-sequence stars (including dwarf stars)
    
    * $VI$
      
      * sub-dwarfs
    
    * $VII$
      
      * white dwarfs

As a helpful reference, the Sun is considered a G2 V star on the main sequence at present. Even without accounting for quantum effects - such as electron degeneracy pressure - there is a dance between gravity pulling the star inwards and thermal pressure pushing the star outwards; when the Sun is a red giant in a few billion years, it will have a different surface temperature, surface area, luminosity, absolute magnitude, spectral classification, and luminosity class. 
<img src="file:///Users/owner/Desktop/programming/hr_diagram/output/example_01-data_via_web_scrape/HR_Diagram-2DHistogram-AbsoluteMagnitude_VS_ColorIndex(BV)-wSun-wIsoRadius-wSpectralTypes_MarkAt9V.png" title="" alt="example-2D_histogram" data-align="center">The data (HYG 4.1) for this project was obtained from the HYG Database, which is provided by Astronomy Nexus; they have compiled data on nearly $120,000$ stars from combined subsets of data from the Hipparcos Catalog, the Yale Bright Star Catalog, and the Gliese Catalog of Nearby Stars.

## Getting Started

### Dependencies

* Python 3.9.6
* numpy == 1.26.4
* matplotlib == 3.9.4
* pandas == 2.2.3
* beautifulsoup4 == 4.13.4 

### Executing program

* Download this repository to your local computer

* Modify `path_to_save_directory` in `src/example_01-data_via_web_scrape.py` and then run the script

* Modify `path_to_save_directory`, `path_to_hyg_data_file`,
  `path_to_classification_file`, and `path_to_data_directory`  in `src/example_02-data_via_file_read.py` and then run the script

## Version History

* 0.1
  * Initial Release

## License

This project is licensed under the Apache License - see the LICENSE file for details.