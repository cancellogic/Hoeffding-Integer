# Hoeffding Dependence Coefficient in Integer Form
This is Wassley Hoeffding's 1948 equation for detecting -frequently nonlinear- relationships in data or variables.  Wassley Hoeffding was there at the founding of modern non-parametric statistics and Hoeffding's work has inspired decades of additional work. 

Hoeffding D = 30 [ (n-2)(n-3)Sum[(Qi-1)(Qi-2)] + Sum[(Ri-1)(Ri-2)(Si-1)(Si-2)] - 2(n-2)Sum[(Ri-2)(Si-2)(Qi-1)] ] / [ n(n-1)(n-2)(n-3)(n-4)] 
        
        where n = number of paired observations, Ri are ranks of first value in pair among all other firsts values with halves for matches  
              Si are ranks of the second value in a pair among all second values with halves for matches
              and Qi are bivariate ranks or in English: a count of how many pairs (as xy points on a plot) are smaller in both paired values (x and y)
              (plus one for self match, plus a quarter for other points that have the same xy values, plus a half when x1=x2 y is lower or y1=y2 x is lower)  

If the above was multiplied by a value such that whole ranks were represented as four quarters, and the Pochhammer [n(n-1)(n-2)(n-3)(n-4)] was factored out
to avoid fractions, the equation would look like:
"H_integer D" = [ 16(n-2)(n-3)Sum[4(Qi-1)4(Qi-2)] + Sum[4(Ri-1)4(Ri-2)4(Si-1)4(Si-2)] - 4*2(n-2)Sum[4(Ri-2)4(Si-2)4(Qi-1)] ]
"H_integer D" = 256 * [ (n-2)(n-3)Sum[(Qi-1)(Qi-2)] + Sum[(Ri-1)(Ri-2)(Si-1)(Si-2)] - 2(n-2)Sum[(Ri-2)(Si-2)(Qi-1)] ]

In this variation of the Dependence calculation, the association is computed with integers by multiplication of the original statistic with 
n(n-1)(n-2)(n-3)(n-4)(256/30) (n=number of pairs).  Higher values have stronger relationships- good for detection of useful models in machine learning or for genetic algorithm fitness assessment especially in non-linear situations. 

Min and Max values of the integer form can be computed for a given length of paired comparisions. 

Coded in Rust for generic partially ordinal types (compare {"g","e","n","e","r","i","c","s"} and {3.0,1.0,4.0,1.0,5.0,4.0,3.5,6.1}).  Yep- directly compare different data types - if they sort they likely can be run through Hoeffding_Integer.

# Why?
Hoeffding's Dependence Coefficient is very goot at assigning fitness to nonlinear models for genetic algorithms and machine learning.  

Why program in RUST?  Rust is fast and has taught me better ways to code!  

Why integer?   A ?fantasy? that integer hoeffding is a step toward GPU computation of Hoeffding's Dependence Coefficient D.  And the annoyance that as n gets large (theoretically leading to higher resolution statistics) the floating point math of the denominator Pochhammer factorial makes small progress invisible (if you have n=1587 pairs, a small scramble of progress is divided by the Pochhammer 10003350094863840 thereby likely vanishing in limited floating point decimals.  

This quibble can be reframed as "what are Hoeffding's D weaknesses?"  In my mind, (1) it is more expensive to compute than Pearsons R (note rust & GPU fantasy) and (2) It finds hidden relationships too well and reports high levels of certainty too quickly for ideal use in selecting "very best of many best" models.  Don't get me wrong - quibbles are quibbles - if you have chemical, biological, behavioral, finance or network data where linear relationships do not rule, please consider Hoeffding D over Pearsons R correlation. 

# The math 
Please read the hundred lines of mathematics in the comments of main.rs or lib.rs!  I've tried to make the statistic's computation possible to verify, follow and understand. 

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

# Babble...
Evygene Slutsky may have anticipated some of Hoeffding's work on dependence, but cold war, stoic man and little published in the west.   I believe ES died the same year H published this statistic... unusual correlation that...
