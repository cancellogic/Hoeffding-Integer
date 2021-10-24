# Integer-Hoeffding
This is Wassley Hoeffding's 1948 equation for detecting -frequently nonlinear- relationships in data or variables.  Hoeffding helped found modern non-parametric statistics and his work has inspired decades of additional work.  
In this version, the dependence cofficient or association value is computed with integer calculation by multiplying Hoeffdings Dependence coefficient D by n(n-1)(n-2)(n-3)(n-4)(256/30) (n=number of pairs).  Higher values do have stronger relationships- good for detection of valuble models in machine learning or genetic algorithm fitness assessment especially in non-linear situations. Coded in Rust for generic partially ordinal types (compare {"g","e","n","e","r","i","c","s"} and {3.0,1.0,4.0,1.0,5.0,4.0,3.5,6.1}).  Yep- directly compare different data types - if they sort they likely can be run through Hoeffding_Integer.
# Why?
Why program in RUST?  Rust is fast.

Why integer - a ?fantasy? that integer hoeffding is a step toward GPU computation of Hoeffding's Dependence Coefficient D.  And the annoyance that as n gets large (theoretically leading to higher resolution statistics) the floating point math of the denominator Pochhammer factorial makes small progress invisible (if you have n=1587 pairs, any floating point "progress" is divided by the Pochhammer 10003350094863840 thereby vanishing small progress in limited floating point decimals.)  
# The math 
Please read the hundred lines of mathematics in the comments of main.rs or lib.rs!  I've tried to make the statistic's computation possible to verify, follow and understand. 
# How to use

first add hoeffding_integer to your cargo file from crates.io (and I haven't published in crates.io as of Oct 2021... but soon)

let textdata: Vec<&str> = vec!["a","a","a","b","b","b","c","c","c"];
let numdata:    Vec<u8> = vec![ 1 , 2 , 3 , 4 , 5 , 5 , 6 , 7 , 8 ];

let hoeffding_dependence_statistic_as_integer:i128 = hoeffding_integer[textdata, numdata];
let dependence_min_max_as_integer:(i128,i128) = hoeffding_integer_min_max( textdata.len() );

println!("Compare text and numbers:  {:?}  vs.  {:?}",&textdata, &numdata);
println!("Hoeffding's dependence coefficient D as integer: {}", hoeffding_dependence_statistic_as_integer );
println!("min and max possible for statistic: {}",dependence_min_max_as_integer);
  
