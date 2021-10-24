//!Hoeffding Dependence Coefficient is good at finding associations, even in many non-linear situations.  For genetic algorithms, it can characterize fitness, 
//!especially where Pearson's correlation R strongly promotes linear solutions in nonlinear problems. This integer version of Hoeffding uses integer representation
//!of half and quarter matches.  Integer use was inspired by the observation that the original normalized function contains a denominator Pochhammer 
//!and progress indicating digits could vanish with floating point conversions for large numbers of pairs (~n>1585).  
//! 
//! Late 1950's version of Hoeffding's Statistic 
//! D = 30 * (   (n-2)(n-3) Sum( (Qi-1)(Qi-2) and Qi is bivariate rank of each XY pair) + Sum((Ri-1)(Ri-2)(Si-1)(Si-2) where Ri and Si are pairs but individual X and Y rank) - 2*(n-2) * Sum((Ri-2)(Si-2)(Qi-1)))    )
//!     /(n(n-1)(n-2)(n-3)(n-4)))
//!  
//!The "Hoeffding Integer" value presented here is the original Hoeffding D coefficent, multiplied by 256/30 * (n)*(n-1)(n-2)(n-3)(n-4).    
//! 
//!Minimun and Maximum possible "Hoeffding Integer" values are offered for a sense of scale.
//! 
//!Please forgive that I did not implement Blum Kieffer and Rosenblatt's 1961 paper or the 2017 "Simplified vs. Easier" papers by Zheng or Pelekis to turn the raw value into a precise probability.  
//!Oye Thar be dragons. The intent for "Hoeffding Integer" use in machine learning is that larger values equal greater probability of associations even if the scale has skew.    
//!  
//!Cheers,  Dustan Doud (September 2021)
//! PS:  the math is documented with comments, although 256 above was a design choice because I represented Qi and Ri and Si all in quarters so (Ri-1)(Ri-2)(Si-1)(Si-2), in quarters is 256 = 4 * 4 * 4 * 4  --> 4(Ri-1) * 4(Ri-2) * 4(Si-1) * 4(Si-2)
// 
use std::fmt::Debug; 

/*INTEGER HOEFFDING STATISTIC- A terrible explainer:  
Imagine asking a friend to randomly place the letters A through Z on graph paper.  But was the placement random?  Maybe they hid something?  To find out without reading, with Hoeffding, first you would get 
the width (x) and height (y) of each letter.  Then you would need to rank the x and y values, followed by ranking their entwined bivariate value.

The x-axis rank for point A would be the count of points to the left of point A (lower x values)
and half the values of points that had exactly same x value as point A.  Much the same for y-axis rank.  0th X or Y ranks are allowed if there is no smaller value.

To compute the XY combined ranks, imagine a x-y axis cross at each lettered point.  Starting with lettered point "A", tally the count of all other points in the lower left (less less) quadrant as +1 point.  
should different lettered point exactly matches on A's x y (equal equal) add one quarter, while lettered points that are (equal less) or (less equal) add a half.   Repeat
tallys for all lettered points A through Z and you'd have your Qi ranks.   Almost.  Statistics and Hoeffding counted bivariate XY ranks and single ranks differently through the early life of this equation (?1948-1957) 
leading toward easy confusion.  In 1948 Hoeffding's ranking system is roughly the answer to the question "how many lessor values exist plus a fraction of how many equal values exist."  So for integers 1 to 10, the rank of 6 would be 5 because five values are less than 6.  
his notion changes, and by 1957 the rank of 6 in 1 to 10 is 6, and the lowest rank can not be 0th anymore but 1st.  So lets not talk about off by one problems in approaching the historic math with a modern perspective 
and just pretend ranking things is easy.
 
Ok.  So if you were to classify ranks for all 26 A-Z letters on your graph paper, you'd have a list of X axis, Y axis and XY ranks called Ri Si and Qi in statistics - (the i is for individual or indexed rank).  
By summing the Ri Si and Qi with a bit of subtraction and multiplication, you can get D1 D2 and D3 statistics that are used to compute a dependence statistic called Hoeffding's D.

And if you did this nightmare homework for an alphabetic set of pairs (or really any set larger than 5) pairs, embedded in the mathematics of the rankings is some indication of dependence, connection and correlation.  
Waisley Hoeffding described this statistical relationship between rankings and dependence or correlation in 1948 as a series of sums, additions and multiplications of the Qi Ri and Si values.  Which makes it sound easy as 
D = 30 * ((n-2)(n-3)*D1 ) + D2 - 2((n-2)D3) /  (n Pochhammer5 factorial) where D1 = Sum [(Qi-1)(Qi-2)], D2 = Sum[(Ri-1)*(Ri-2)(Si-1)(Si-2)] and D3 = Sum[(Ri-2)(Si-2)(Qi-1)].
So if your friends 26 letters spilled across the page spelling out "UNCOPYRIGHTABLE" (a word with 15 unique letters) in a clear wave, this equation might well flag that it found an association.   

*/

fn hoeffding_integer_min_max(number_of_pairs:usize) -> (i128,i128) {
    let n:i128 = number_of_pairs as i128;
   /*
   As I imagine this function, maximum statistic should be where ranks are perfectly nested russian doll box style and minimum values should be where values and ranks are equal.  (that's not a proof, this function accepts generics so theoretical comparisons might include (Not a Number < Negative Infity)? and other oddities) 
   Max association with (x,y) = {(1 1),(2 2),(3 3),(4 4),(5 5)} or similar and min with x,y = {(0 0),(0 0),(0 0),(0 0),(0 0)} resulting in min rank all equal to n/4 + 3/4.  
   So Ri=Si=Qi in both max and min, but one is flat and the other steps evenly upward.   
     
   Let start with compute of D1 D2 D3 summations for number n in perfectly nested russian doll boxes or ranks.  (i.e. if there are 5 ranked x and y items then n=5, presume x ranks of {0 1 2 3 4} y ranks of {0 1 2 3 4} and xy ranks of {1 2 3 4 5} for perfect nesting.  (aside: yes xy ranks appear to use +1 scale vs x and y in 1957, no I can't tell if that is perfectly correct.  Guessing it isn't perfect because "improved" Hoeffding statistics computation methods have been researched for decades)  
   Presume a perfectly nested set of all three ranks results in a maximum D1 value.  Presume all rank values are multipled by 4"quarters" and compute max D1, D2, D3.
   D1 cis Sum[(Qi-1)(Qi-2), Qi, 1, n] where Qi represents XY ranks is rearranged into Sum[(4i-4)(4i-8),i,1,n] x 1/16_ignore_for_now) yielding output via WolframAlpha  
   Sum[(4i - 4)(4i-8),{i,1,n}] --> 16/3 * n * (n*n - 3*n +2) --> 16 * n * (n-1) * (n-2) / 3   
   **also note that (n)*(n-1)*(n-2) will always be evenly divisible by 3 for integer n because either n, n-1 or n-2 must be evenly divisible by 3**

   D1_max_perfectly_nested_ranks_times16 = n * (n*n - 3*n + 2) * 16 / 3
   
   Moving on to maximum D2,
   D2 = Sum((Ri-1)(Ri-2)(Si-1)(Si-2)) but R and S are in integer quarters so D2_max = (Sum(( 4Ri-4)(4Si-4)(4Ri-8)(4Si-8) )) ) (x 1/256_ignore_for_now) 
   Given as input to WolframAlpha as Sum[(4i-4)(4i-4)(4i-8)(4i-8), {i,1,n}] yeilding output
   D2_max_times256 = 256 * n(3n*n*n*n - 15n*n*n + 25 n*n - 15N +2) / 15 
    Which turns out to always be evenly divisible by 15 for integer n because mods 3/15 - 15/15 + 25/15 - 15/15 + 2/15 exactly equal 0. 

   --> Simplify[256 * n(3n*n*n*n - 15n*n*n + 25n*n - 15n +2)/15] -->
   D2_max_times256 = 256 * n*(n-1)*(n-2)*( n * (3n-6) +1) / 15.  
 
   And finally lets calculate the D3_maximum_value,
   D3 = Sum[(Ri-2)(Si-2)(Qi-1), for all i] but we have integer quarter representation and Ri=Si=Qi for nested maximum ranks so 
   D3_max_times64 = Sum[(4i-8)(4i-8)(4i-4), {i,1,n}] which can be solved by Wolfram Alpha as equal to
   --> 16 * n * (3 * n * n * n - 14n*n + 21n - 10) /3 --> Simplify[Sum[(4i-8)(4i-8)(4i-4), {i,1,n}] ] -->  16 * n * (n-1) * (n-2) * (3n-5) / 3
   which also happens to be evenly diviible by three because of the embeded pochhammer (n)(n-1)(n-2) must contain a factor of 3 (if n isn't n-1 or n-2 must be) 

  D = 30 (n-2)(n-3)*D1 + D2 - 2(n-2)* D3  /  (n(n-1)(n-2)(n-3)(n-4))
  D_times_pochhammer = (n-2)(n-3)*D1 + D2 - 2(n-2)* D3 
  D_max_times_pochhammer_times_256 = (n-2)*(n-3)* 16_multiplier * {{D1 = n * (n*n - 3*n + 2) * 16 / 3}} +
                                     {{D2 = 256 * n(3n*n*n*n - 15n*n*n + 25 n*n - 15N +2) / 15  }} -
                                      2*(n-2)*{{d3=  4_multiplier * 16 * n * (n-1) * (n-2) * (3n-5) / 3}}
 
   in Wolfram Alpha:> Simplify[ (n-2)*(n-3)* 16 * ( n * (n*n - 3*n + 2) * 16 / 3) +   (256 * n(3n*n*n*n - 15n*n*n + 25 n*n - 15n +2) / 15 ) -  2*(n-2)*( 4 * 16 * n * (n-1) * (n-2) * (3n-5) / 3) ]
 
   D_max_times_pochhammer_times256 = 128/15 * n * (n-1) * (n-2) * (n-3) * (n-4)
   (once again no fraction because n*(n-1)*(n-2) must be divible by 3 for integer n, and (n)(n-1)*(n-2)*(n-3)(n-4) likewise must contain a value divible by 5)
    */
    let hoeffding_integer_maximum:i128 = (128  * n * (n-1) * (n-2) * (n-3) * (n-4) ) / 15;
/*  
    And on to slightly simplier hoeffding_integer_MINIMUM values:

    Here I imagine minimum values to exist when all matches are equal quarter matches, except the match to self worth a whole.  
    Ri = Si = Qi and each individual rank has a value of 3/4 + n/4.   
    Wolfram Alpha :>  Simplify[   Sum[ (4(3/4 + n/4) - 4)((3/4 + n/4)-8),{i,1,n}]  ]   
    D1_min_times16 = n(n-1)(n-5)
    
    On to D2
    Wolfram Alpha :>Simplify[ Sum[(4(3/4 + n/4) - 4)(4(3/4 + n/4)-8)(4(3/4 + n/4) - 4)(4(3/4 + n/4)-8),{i,1,n}     ] ]
    D2_min_times_256 = n * (n-1)*(n-1) * (n-5)*(n-5)
   
    And lastly D3
    Wolfram Alpha :> Simplify[ Sum[(4(3/4 + n/4) - 4)(4(3/4 + n/4)-8)(4(3/4 + n/4)-8),{i,1,n}     ] ]
    D3_min_times64 = PLEASE CONSIDER PURCHASING A PRO ACCOUNT WITH WOLFRAM ALPHA :)
                   = n * (n-1) * (n-5) * (n-5)

    D_minimum_times_pochhammer_times_256 = ( (n-2)(n-3) *  16 * n(n-1)(n-5) ) + 
                                            n * (n-1)*(n-1) * (n-5)*(n-5) - 
                                             2*(n-2) *     4 * n * (n-1) * (n-5) * (n-5)

     Wolfram_Alpha :> Simplify[( (n-2)(n-3) *  16 * n(n-1)(n-5) ) + n * (n-1)*(n-1) * (n-5)*(n-5) -  2*(n-2) * 4 * n * (n-1) * (n-5) * (n-5)   
                     = 3n(n-1)(n-1)(n-5)(3n-7)                                        
    */
    let hoeffding_integer_minimum:i128 = 3* n * (n-1) * (n-1) * (n-5) * (3*n - 7);

(hoeffding_integer_maximum, hoeffding_integer_minimum)

}

fn order_sort_by<T: Clone + Copy + PartialOrd >(list: &Vec<T>) -> Vec<usize> {
    //(1) Create index
    let mut orders: Vec<usize> = (0..(list.len())).collect();

orders.sort_by( |a,b| (list[*a]).partial_cmp(&list[*b]).unwrap());

    return orders;
}

fn count_sorted_duplicates_or_uniques<T: PartialOrd + Clone>(splice: &Vec<T>) -> Vec<usize> {
    let length = splice.len();
    let mut count: usize = 1;
    let mut output: Vec<usize> = vec![];

    
        for i in 0..(length - 1) {
            if splice[i] == splice[i + 1] {
                count += 1;
            } else {
                output.push(count);
                count = 1;
            }
        }
    
            output.push(count);
        

        output
    
}

fn hoeffding_integer<T: PartialOrd + Clone + Copy + Debug, P:PartialOrd + Clone + Copy + Debug >(datax: &Vec<T>, datay:&Vec<P>) -> i128{

    //globals
    let n_element = datax.len();
    match datax.len() == datay.len() {
     false => {return -1},
     true => {
         if datax.len()<5 {eprint!("Hoeffding Dependence Coefficient requires 5 pairs or more (5 degrees of freedom) least the inverse 5'Pochhammer factorial within grasp infinity"); return -2}
     },
     };

    
    //Start with ranking X data
    let arrangementx:Vec<usize> = order_sort_by(&datax);
    let repatterx: Vec<T> = arrangementx.iter().map(|v| datax[*v]).collect();
    let countedx: Vec<usize> = count_sorted_duplicates_or_uniques(&repatterx);
    let uniques_lengthx = countedx.len();

    let mut runningtotalx = 1;
    let mut outgoingx: Vec<usize> = vec![];

    for each in 0usize..uniques_lengthx {
        runningtotalx += countedx[each];
        let wholesx: usize = runningtotalx.saturating_sub( countedx[each] );
        let halvesx:usize = countedx[each].saturating_sub(1);
        let quadrankx = 4*wholesx + 2*halvesx;
        for _every in 0..countedx[each] {
              outgoingx.push(quadrankx.clone());
        }
    }

    let mut originx:Vec<usize> = vec![0usize ; datax.len()];
    for each in 0..(outgoingx.len())  {
        originx[arrangementx[each]] =  outgoingx[each]
    }
    
  //then rank y data

    let arrangementy:Vec<usize> = order_sort_by(&datay);
    let repattery: Vec<P> = arrangementy.iter().map(|v| datay[*v]).collect();
    let countedy: Vec<usize> = count_sorted_duplicates_or_uniques(&repattery);
    let uniques_lengthy = countedy.len();
    
    let mut runningtotaly = 1;
    let mut outgoingy: Vec<usize> = vec![];

    for each in 0usize..uniques_lengthy {
        runningtotaly += countedy[each];
        let wholesy: usize = runningtotaly.saturating_sub( countedy[each] );
        let halvesy:usize = countedy[each].saturating_sub(1);
        let quadranky = 4*wholesy + 2*halvesy;
        for _every in 0..countedy[each] {
              outgoingy.push(quadranky.clone());
        }
    }

    let mut originy:Vec<usize> = vec![0usize ; datay.len()];
    for each in 0..(outgoingy.len())  {
        originy[arrangementy[each]] =  outgoingy[each]
    }

  
    // and now on to the tedious rank of xy data where being greater in X & Y counts for 4 points, 
    // being greater on 1 axis and matched counts for 2 points, and matching each axis counts for 1 point of rank 
   let mut originxy:Vec<usize> = vec![];

   for j in 0..n_element { //wish there was a less insanity laden path than n^2 - but majority of paried x&y data comparisons will not be be deftly organized, low noise and strongly correlated.
     let v:usize =   (0..n_element).map(|i|  2*((originx[i]<originx[j] && originy[i]<originy[j] )as usize) + 
                                                          2*((originx[i]<=originx[j] && originy[i]<=originy[j]) as usize) -
                                                          ((originx[i]==originx[j] && originy[i]==originy[j]) as usize)   
                    ).collect::<Vec<usize>>().iter().sum();
                    let v=v+3;
       originxy.push(v);            //
   }   
      
   // and you'll note that x y and xy (Ri Si Qi) ranks are all represented by quarters at this point
    //D1 = Sum( (Qi-1)(Qi-2) ) but Qi is represented in integer quarters so D1 = (Sum( (4Qi - 4)(4Qi-8) ))/16 
    let d_subone_times_sixteen_list:Vec<i128> = (0..n_element).map(|i|  ((originxy[i] as i128) - 4) * (((originxy[i] as i128)) - 8  )).collect();  //note that originxy[i] = Q_sub_i - 1 (bivariate rank minus 1 for each i)
    let d_subone_times_sixteen:i128 = d_subone_times_sixteen_list.iter().sum();
    println!("d1={}",&d_subone_times_sixteen);

    //D2 = Sum((Ri-1)(Ri-2)(Si-1)(Si-2)) but R and S are in integer quarters so D2 = (Sum(( 4Ri-4)(4Si-4)(4Ri-8)(4Si-8) )) )/256
    let d_subtwo_times_twofiftysix_list:Vec<i128>= (0..n_element).map(|i| ((originx[i] as i128) -4) * ((originx[i] as i128)-8) * ((originy[i] as i128)-4)*((originy[i] as i128)-8)).collect() ;
    let d_subtwo_times_twofiftysix:i128 = d_subtwo_times_twofiftysix_list.iter().sum();
    println!("d2={}",&d_subtwo_times_twofiftysix);
    
    //D3 = Sum((R-2)(S-2)(Q-1)) but integer quarters again so D3 = Sum ((4Ri-8)(4Si-8)(4Qi-4)) / 64
    let d_subthree_times_sixtyfour_list:Vec<i128> = (0..n_element).map(|i| ((originx[i] as i128)-8i128 )*((originy[i] as i128)-8i128)*((originxy[i]as i128 )-4i128)  ).collect();
    let d_subthree_times_sixtyfour: i128 = d_subthree_times_sixtyfour_list.iter().sum();
    let n = n_element as i128;
    println!("d3={}",&d_subthree_times_sixtyfour);

    // D = 30* ((n-2)*(n-3)*D1 + D2 - 2*(n-2)*D3 ) / (n*(n-1)*(n-2)*(n-3)*(n-4)) 
    //lets multiply 30 about by 256/30 so we can avoid pesky fractions in integer calculation
    //let h_first_times256:i128 = (256/16 = 16) *(( ((n-2)*(n-3)) * d_subone_times_sixteen) as i128);  //n is greater than 4 and therefore 30*(n-2)*(n-3) is always positive and evenly divible by four.  
    let h_first_times256:i128 = 16 *(( ((n-2)*(n-3)) * d_subone_times_sixteen) );
    println!("head1{:?}",&h_first_times256);

    //let h_second_times256:i128 = 256/256 *(d_subtwo_times_twofiftysix ) as i128;
    let h_second_times256:i128 =  (d_subtwo_times_twofiftysix ) ;
    println!("head2{:?}",&h_second_times256);

    //let h_third_times256:i128 = 256/64* (2n-4)*d_subthree_times_sixtyfour) as i128; //note (2n-4) becomes n-2 to account for multiply by two
    let h_third_times256:i128 = 4 * (2 * n-4) *d_subthree_times_sixtyfour; //note (2n-4) becomes n-2 to account for multiply by two
    println!("head2{:?}",&h_third_times256);
   
    let  hoeffding_integer_numerator:i128 = h_first_times256 + h_second_times256 - h_third_times256;

hoeffding_integer_numerator // linear to original statistic
// denominator = ( (256 * n * (n-1) * (n-2) *(n-3) * (n-4)) / 30)
// And lets not divide or bother with normalization or pochhammer because I'm (unreasonably) trying to keep hundreds of siginificant digits and want to avoid floating point representations.  

}

fn rank_on_climbing_sorted_data_value_doubled<T: PartialOrd + Clone + Copy + Debug >(data: &Vec<T>) -> Vec<usize> {
    let arrangement: Vec<usize> = order_sort_by(&data);
    let repatter: Vec<T> = arrangement.iter().map(|v| data[*v]).collect();
    let counted: Vec<usize> = count_sorted_duplicates_or_uniques(&repatter);
    let uniques_length = counted.len();
    //let mut prior = counted[0];
    let mut runningtotal = 0;
    let mut outgoing: Vec<usize> = vec![];

    for each in 0usize..uniques_length {
        
        runningtotal += counted[each];
        let wholes: usize = runningtotal.saturating_sub( counted[each] );
        
        let halves:usize = counted[each].saturating_sub(1);
        
        let doubledrank = 2*wholes + halves;
        
        for _every in 0..counted[each] {
              outgoing.push(doubledrank.clone());
        }
        //prior = runningtotal;
    }

    let mut origin:Vec<usize> = vec![0usize ; data.len()];
    for each in 0..(outgoing.len())  {
        origin[arrangement[each]] =  outgoing[each]
    }

    origin
}

fn main() {
    //let data: Vec<&str> = vec!["a","a","a","b","b","b","c","c","c"];
    //let data2: Vec<&str> = vec!["a","b","c","a","b","c","a","c","c"];
   let data:Vec<u8>= vec![1,2,3,4,5,6,7,8,9,10];
   let data2:Vec<u8> = vec![2,3,4,5,6,7,8,9,10,12];
   //let data2: Vec<&str> = vec!["a","b","c","d","e","f","g","h", "i", "j"];
//let now1 = std::time::Instant::now();
let d = hoeffding_integer(&data, &data2);//let hoeff = hoeffding_dependence_coefficient(&longdataa, &longdatab);
println!("statistic {}",d);
let (min,max) = hoe
println!("min and max: {:?}",)
//let now2= std::time::Instant::now(); let timekeeper = now2-now1;

}
