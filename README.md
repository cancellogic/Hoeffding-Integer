# Hoeffding Dependence Coefficient in Integer Form:  A good way to find relationships in data
This is Wassley Hoeffding's 1948 equation for detecting -frequently nonlinear- correlation, coincidence or connections in data, but presented as an integer between hypothetical minimum and maximum - rather than in the decimal form where 1.0 is maximum correlation and every #n has a unique minimum.   Use Hoeffding for AI and data eureka moments.  

Wassley Hoeffding was there at the founding of modern non-parametric statistics and Hoeffding's work has inspired decades of additional work. And I' ve touched his work by multiplication (D times 256 * five_term_Pochhammer_of_n / 30 ) => so lets go over the details:

Hoeffding D = 30 [ (n-2)(n-3)Sum[(Qi-1)(Qi-2)] + Sum[(Ri-1)(Ri-2)(Si-1)(Si-2)] - 2(n-2)Sum[(Ri-2)(Si-2)(Qi-1)] ] / [ n(n-1)(n-2)(n-3)(n-4)] 
        
        where D = Dependency, correlation, connection, association, or how unlikely chance might align paired lists.
              n = number of paired observations, 
              Ri are ranks of first value in pair among all other firsts values with half awarded for matches.  (lowest rank is 0 in 1948 and 1 in late 1950's)   
              Si are ranks of the second value in a pair among all second values with half awarded for matches. 
              and Qi are bivariate ranks or in English: Systematically index each pair and plot as xy points. To rank - start at 3/4 for a given pair, 
              and then count all other pairs that are lower/left of the given pair and add 1 point.   Pairs match on x or y but are less on opposite axis get
              +1/2 points.  And duplicates of the given pair are worth 1/4 point (and include self - the given point - because 1/4 + 3/4 start = 1 whole
              self.)  Continue through the index of all pairs untill all points are ranked and you have the Qi.

If Hoeffding's Dependence was multiplied by a value such that whole ranks were represented as four quarters, and the Pochhammer [n(n-1)(n-2)(n-3)(n-4)] was factored to avoid fractions, the equation would look like:

        "H_integer D" = [ 16(n-2)(n-3)Sum[4(Qi-1)4(Qi-2)] + Sum[4(Ri-1)4(Ri-2)4(Si-1)4(Si-2)] - 4*2(n-2)Sum[4(Ri-2)4(Si-2)4(Qi-1)] ]
        simplified-->
        "H_integer D" = 256 * [ (n-2)(n-3)Sum[(Qi-1)(Qi-2)] + Sum[(Ri-1)(Ri-2)(Si-1)(Si-2)] - 2(n-2)Sum[(Ri-2)(Si-2)(Qi-1)] ]

In this variation of the Dependence calculation, the association is computed with integers by multiplication of the original statistic with 
n(n-1)(n-2)(n-3)(n-4)(256/30) (n=number of pairs).  Higher values have stronger relationships- good for detection of useful models in machine learning or for genetic algorithm fitness assessment especially in non-linear situations. 

Min and Max values of the integer form can be computed for a given length of paired comparisions.  Maximum assumes unique, growing, non-repeating values in every pair, but unique values may not be possible when comparing yes/no or 8bit data where the number of allowed states is less than the number of pairs and repeats abound.  Binary comparisions may not even be a correct use, but they do suggest to me the Hoeffding's dependence form could be pivoted into a form with bit bin tallys for fast computation and far less sorting CPU time (and thar be dragons that eat manhours).   So... this hoeffding_integer_d_maximum doesn't understand your specific data constraints- and you may need to write a situational maximum function or adjust goal endpoints accordingly.  

Coded in Rust for generic partially ordinal types (compare {"g","e","n","e","r","i","c","s"} and {3.0,1.0,4.0,1.0,5.0,4.0,3.5,6.1}).  Yep- directly compare vastly different data types - if they sort they likely can be run through Hoeffding_Integer.  For the data scientist that flexability is beautiful- for the compiler and programmer- my condolences on extra seconds to enjoy a sip of a hot beverage. 

# Why?
Hoeffding's Dependence Coefficient is very good at assigning fitness to nonlinear models for genetic algorithms and machine learning.  The Pochhammer free version isn't normalized so direct comparision between n=50 and n=1000 is not clear even with min and max values for each.  But this still works well in machine learning when models have "not-a-number" defects (defects automatically get lower scores if more than one defect exists, and automatically get lower scores should invalid values be pruned.)

Why program in RUST?  Rust is fast and has taught me better ways to code!  And crates like Rayon let this single thread function run on every cpu core ("only once" in machine learning might not be a thing.) 

Why integer?   A ?fantasy? that integer hoeffding is a step toward GPU computation of Hoeffding's Dependence Coefficient D.  And the annoyance that as n gets large (theoretically leading to higher resolution statistics) the floating point math of the denominator Pochhammer factorial makes small progress invisible (example:  if you have n=1587 pairs, a small scramble of progress is divided by the Pochhammer 10003350094863840, mayhaps vanishing progress in limited floating point decimals.)  

This quibble can be reframed as "what are Hoeffding's D weaknesses?"  In my mind, (1) it is more expensive to compute than Pearsons R (note rust & GPU fantasy) and (2) It finds hidden relationships too well and reports high levels of certainty too quickly for ideal use in selecting "very best of many best" models.  Don't get me wrong - quibbles are quibbles - if you have chemical, biological, behavioral, finance or network data where linear relationships do not rule, please consider Hoeffding D over Pearsons R correlation. 

# The math 
Please read the hundred lines of mathematics in the comments of main.rs or lib.rs!  I've tried to make the statistic's computation possible to verify, follow and understand.    The TLDR route from Hoeffding D to Hoeffding integer is multiply the orginal equation by 256 times the n!5'Pochhammer all divided by 30.  

# How to use
first add hoeffding_integer to your cargo file from crates.io (Published late Oct 2021)

    let textdata: Vec<&str> = vec!["a","a","a","b","b","b","c","c","c"]
    let numdata:    Vec<u8> = vec![ 1 , 2 , 3 , 4 , 5 , 5 , 6 , 7 , 8 ];

    let hoeffding_dependence_statistic_as_integer:i128 = hoeffding_integer[textdata, numdata];
    let dependence_min = hoeffding_integer_minimum( textdata.len() );
    let dependence_max = hoeffding_integer_maximum( textdata.len() );

    println!("Compare text and numbers:  {:?}  vs.  {:?}",&textdata, &numdata);
    println!("Hoeffding's dependence coefficient D as integer: {}", hoeffding_dependence_statistic_as_integer );
    println!("min and max possible for statistic: {} <--> {}",dependence_min, dependence_max);

# How fast or slow?  --release and benchmarks   Random sample?
First, be sure to "cargo run --release", the optimized release builds are so good that it feels like a bug not to use them. 

For many reasons, I love & endorse Wolfram's computational products like Mathematica and WolframAlpha!  Single thread results are reported for one thousand and ten thousand random pairs. Compared to Mathematica 12.2 on a 32bit os Raspberry Pi 4, for n=1000 Hoeffding_integer_d (with Rust 1.56.1 built as "cargo build --release") ran ~5.6 times faster than Mathematica, while for n=10000 this code was only 1% faster.  Moving to a Intel Xeon with "WolframKernel 12.3.1", Hoeffding_integer_d runs n=1000 random pairs 22x faster than WolframKernal and n=10000 random pairs runs 4.3x the speed WolframKernal.  Your speed may vary with Rayon multicore use, 32bit vs 64bit os, hardware, number of pairs, data configuration (presorted vs unsorted), number of cpu threads active, etc. 

Last speed tips:  Run small (<10k) to run fast - this function slows roughly by the cube of pairs.  Exclude missing data.  You probably don't need to exactly compute D in order to make a discovery.  Focus - if you are looking for rare events in a sea of normal data, randomly exclude "normal" pairs.   For large datasets (over 60k?) consider random subsampling to reduce #n pairs - unless subsampling would exclude "rare" data.  
Goodluck!        

# Babble...
Evygene Slutsky seems under appreciated and yet may have anticipated some of Hoeffding's work on dependence, as well as Lorenz's work on attractors-- but cold war, stoic man, untidy dawn of idea that equations from politics would be applicable to economics or physics because they are statistics and little published in the west.   I believe ES died the same year H published this statistic... unusual correlation that...
