//!Hoeffding Dependence Coefficient is good at finding associations, and may be useful to characterize health in nonlinear genetic algorithm equation modeling, especially where Pearson's correlation R  
//!strongly promotes linear solutions to nonlinear problems. This version of Hoeffding uses ?faster? interger, not floating point, representation of halves and
//!quarters.  Please forgive that I did not implement Blum Kieffer and Rosenblatt's 1961 paper or the 2017 "Simplified vs. Easier" papers by Zheng or Pelekis 
//!that turn the raw coefficient into a probability.  Oye Thar be dragons. The intent for machine learning is that larger values equal greater probability of associations even if the scale isn't between {-0.5 and 1}.    
//!
//!Dustan Doud (September 2021)
//!note(1):  Hoeffding Coefficient can't be solved without at least 5 pairs.  Its a degree of freedom thing.  In those situations consider Pearsons correlation r... or just avoid those situations entirely.    
//!note(2):  Max and min tables are pre-computed for n=5..3005 pairs.  There will be a delay for larger values.  
//!note(3):  Consider random sampling subsets to speed computation unless you seek the very rare or fear throwing away any signal.
//!note(4):  The interger math probably overflows/fails around thirteen million paired elements.  And if you must work with millions of pairs -> this algorithm is painfully slow owing to repeat internal sorts and ranking.
//!          If you need billions of pairwise comparisions and yet do not fear calculation interupting power loss or system reboot because erlang or something,  Crates.io "fdec" macro can build a custom i160 or i256 type to globally replace i128 types in this library.  
//!note(5):  RE:  github/tnagler/wdm/blob/master/include/wdm/hoeffd.hpp estimates Hoeffdings B value and approximate p values by interpolation of 4 decimal place tabulated values.  Nagler's math is almost certainly excellent - 

// #[allow(unused_imports)]  //leftover from  pre-release tests
use std::fmt::Debug;


/*NUMERATOR ONLY HOEFFDING STATISTIC- A simplified explainer:  
Imagine randomly placing the letters A through Z on graph paper.  But was your placement random?  To find out with Hoeffding, first you would measure the width (x) and height (y) of each letter.  Then you would need to rank the x and y values.

The x-axis rank for point A would be the count of points to the left of point A (lower x values)
and half the values of points that had exactly same x value as point A.  Much the same for y-axis rank.  0th X or Y ranks are allowed if there is exactly one minimum value.

The XY combined ranks would be the count of lettered points both below and to the left on the graph (lower x AND y pairs, excluding any pairs that were larger or equal in one or both axis)  And you'd add a quarter for exact matches (where multiple lettered points had the same coordinate - 
give a quarter because the point is circled only by a quarter of the bounding box) and add half for lettered points that were the same on one axis but lower on the other (for a point a edge, but not corner, half of the smallest circle you could draw would still be outside of the bounding box).  
And Hoeffding counts self-matches for XY too, so add 1 whole to all XY ranks... at least in 1957.  

If you were to repeat the process for all 26 A-Z letters on your graph paper, you'd have a list of X, Y and XY ranks called Ri Si and Qi in statistics - (the i is for individual or indexed rank).

If you did this nightmare homework for an alphabetic set of pairs (or really any set larger than 5) pairs, embedded in the mathematics of the rankings is some indication of dependence, connection and correlation..    Waisley Hoeffding described this statistical relationship between 
rankings and dependence or correlation in 1948 as a series of sums, additions and multiplications of the Qi Ri and Si values.  Which makes it sound easy as D = 30 * ((n-2)(n-3)*D1 ) + D2 - 2((n-2)D3) /  (n Pochhammer5 factorial) where D1 = Sum [(Qi-1)(Qi-2)], D2 = Sum[(Ri-1)*(Ri-2)(Si-1)(Si-2)] 
and D3 = Sum[(Ri-2)(Si-2)(Qi-1)]... and add a tiny trip through infinite sequence monticarlo game theory and you'd get a probability of dependence or connection between the plotted Letters.  And while all that might spot letters lined up in a SineWave spelling "UNCOPYRIGHTABLE".. it wouldn't actually read the letters.  It isn't that
kind of algorithm.   
 */

fn compute_sums_for_min_max_n(number:usize) -> i128{
   /*
   compute D1 D2 D3 summations for number n in perfectly nested russian doll boxes or ranks.  (i.e. if there are 5 ranked x and y items then n=5, presume x ranks of {0 1 2 3 4} y ranks of {0 1 2 3 4} and xy ranks of {1 2 3 4 5} for perfect nesting.  (aside: yes xy ranks appear to use +1 scale vs x and y in 1957, no I can't tell if that is perfectly correct.  Guessing it isn't perfect because "improved" Hoeffding statistics computation methods have been researched for decades)  
   Presume a perfectly nested set of all three ranks results in a maximum D1 value.  Presume all rank values are multipled by 4"quarters" and compute max D1, D2, D3.
   D1 computation starting with Sum[(Qi-1)(Qi-2), Qi, 1, n] where Qi represents XY ranks is rearranged into Sum[4(n-1)(n-2),n,1,n] yielding output via WolframAlpha  
   D1_perfectly_nested_ranks =  (4 * n * (n-1) * (n-2) ) / 3   And note that the Pochhammer 3 factorial means integer result will always be evenly divisible by 3 because if n was not divisible by 3, certainly multipliers n-1 or n-2 would be.

   D2 computation starting at Sum[(Ri-1)(Ri-2)(Si-1)(Si-2),0,n-1] ] where Ri=X ranks and Si =Y ranks and X ranks and Y ranks are presumed matched and perfectly nested so Ri=Si=n and multipled by 4"quarters" yielding Sum[4(n-1)(n-2)(n-1)(n-2)],0,n-1] which computes by WolframAlpha into
   D2_perfectly_nested_ranks =  ( 4*n*(182 - 210n + 115n*n - 30n*n*n + 3 n*n*n*n) )/15 and note that the numerator is always evenly divisible by 15 because (182 mod 15 = 2) - (115 mod 15 = 0) + (210 mod 15 = 10) - (30 mod 15 = 0) + (3 mod 15 = 3) yields a remainder of 15 (divisible by 15) for any integer n.  Yep.. I glossed over bits on this proof.
                                        
   D3_perfectly_nested_ranks starts with Sum[(Ri-2)(Si-2)(Qi-1),1,n]... X ranks = Y ranks but != XYranks due to change in ranking scale (starting at 0 vs starting at 1) so substitute (n-1) for Ri,Si and sub n for Qi yielding Sum[4*((n-1)-2)((n-1)-2)(n-1), 1, n] evaluated by WolframAlpha  
   D3_perfectly_nested_ranks = (-32n + 51n*n - 22n*n*n + 3n*n*n*n)/3 and note numerator always divisible by 3 because (-32 mod 3 = -2) + (51 mod 3 =0) - (22 mod 3 = -1) + (3 mod 3 = 0) = -3 which is always evenly divisble by 3 for any integer n
   
   And the maximum values for D1 D2 and D3 are completed.   :)  The math is here is *cough* rough, but the finished form is simple enough to quickly compute. 
    
   The Minimal value should be given in the situation where all samples are the same (equal to zero) yielding only quarter matches across all X Y and XY ranks... strictly speaking this isn't always an allowed data pattern but to complete the math I'm pretending we don't know any better.    
   so for n.len()=9 items the Ri and Si would each have eight quarter matches worth 1 ranking in at (2 2 2 2 2 2 2 2 2) on a whole range or (8 8 8 8 8 8 8 8 8) on a quarter range.  
   The Qi matches self wholely (four quarters) and then adds a quarter per matched pair if all nine are the same yielding (3 3 3 3 3 3 3 3 3) or (12 12 12...) on a quarter range (and... I complain that this looks like an off by one error fixed downstream by Hoeffding, but I'm well outside my field dabbling in number theory and statistics and clearly he was brilliant.)  
   So every Qi_quarter_scale = (n.len() + 3) and D1_min = Sum[(Qi-4"quarters")(Qi-8"quarters") for all i], where all i can be represented by Qi.len()=n.len() and therefore can be arranged as: 
   D1_min =  n.len() * ( ((n.len()+3) - 4)((n.len()+3) - 8) )     where n.len() represents the number of pairs, not a sequence of counted index values for n... and that can be simplified to
   D1_min = n*(n-1)*(n-5)   where n = number of pairs
   
   For D2_minimum, Ri and Si are taken to all equal n.len()-1. So Sum[ ((n.len()-1)-4"quarters")((n.len()-1)8"quarters"))^2 for all items] becomes:
   D2_minimum = n.len() *  ((n.len()-1)-4"quarters")((n.len()-1)-8"quarters"))^2     where n.len() represents the number of pairs, not a sequence of counted index values for n.  And that simplifies to the merely unsightly
   D2_minimum = n*(n-5)(n-5)*(n-9)(n-9)    where n = number of pairs

   For D3_minimum, Ri and Si are taken to all be equal to n.len()-1.  Qi are taken to all equal n.len()+3.  So Sum[ (((n.len() - 1) - 8"quarters")^2) * ((n.len()+3)- 4"quarters") for all items ]
   D3_minimum = n.len() * ( (((n.len() - 1) - 8"quarters")^2) * ((n.len()+3)- 4"quarters") )  which can be further simplifed to
   D3_minimum = n * (n-1) * (n-9) * (n-9)

   And those D1 D2 and D3 get used to make D where

   D_min = 30 * ( (n-2)(n-3)D1 + D2 - 2(n-2)D3 ) / (n(n-1)(n-2)(n-3)(n-4))  So having solved D1 D2 and D3 for n, we can make the substitutions and get this moderately unloveable equation
   D_min = 30 * [(n-2)(n-3){n*(n-1)(n-5)} + {n*(n-5)*(n-5)*(n-9)*(n-9)}   + 2 * (n-2) * {n*(n-1)*(n-9)*(n-9)}] / (n * (n-1)*(n-2)*(n-3)*(n-4) )    which can be simplifed with WolframAlpha to the merely quixotic 
   D_min = 120 + 1595/(n-4) - 4320 / (n-3) + 6615/(n-2) - 5120 / (n-1) 
   and this simplified form of D shows that for nearly every value of n>5 D_min will not be an integer.  But... what if instead of floating point representation we settled on a "interger no mater what" philosophy.  D, the source of Dependence doesn't really require
   a more or less static denominator *if* one is always making comparisons between the same number of items and if one can make due with better or worse rather than a more precise sense of probability.  So for Hoeffding_Integer
let us multiple D_min and D_max by (n)(n-1)(n-2)(n-3)(n-4)/30 
  D_interger_numerator_min = [(n-2)(n-3){n*(n-1)(n-5)} + {n*(n-5)*(n-5)*(n-9)*(n-9)}   + 2 * (n-2) * {n*(n-1)*(n-9)*(n-9)}]    

   */
   let n = number as i128;
   let d_min_times_16 = 16*120 + 16*1595/(n-4) - 16*4320 / (n-3) + 16*6615/(n-2) - 16*5120 /(n-1);
   let d_min_mod = (d_min_times_16 % 16) / 8;  //round up to nearest integer
   let d_min = d_min_mod + d_min_times_16/16;

d_min



}

fn rank<T: Clone + Copy + PartialOrd >(list: &Vec<T>) -> Vec<usize> {
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
//fn fourequity(a:&usize, b:&usize, x:&usize, y:&usize)->usize{


//    let out:usize = 2usize * ((*a>*b) as usize) * 2 * ( (*x>*y) as usize) -(1* ( (*a == *b) as usize) * (*x==*y as usize); 
    //return out
//}
fn hoeffding_health<T: PartialOrd + Clone + Copy + Debug, P:PartialOrd + Clone + Copy + Debug >(datax: &Vec<T>, datay:&Vec<P>) -> i128{

    //globals
    let n_element = datax.len();
    match datax.len() == datay.len() {
     false => {return -1},
     true => {
         if datax.len()<5 {eprint!("Poached Hoeffding Dependence Coefficient requires 5 degrees of freedom (5 paired values or more...)"); return -2}
     },
     };

    
    //Start with ranking X data
    let arrangementx:Vec<usize> = rank(&datax);
    let repatterx: Vec<T> = arrangementx.iter().map(|v| datax[*v]).collect();
    let countedx: Vec<usize> = count_sorted_duplicates_or_uniques(&repatterx);
    let uniques_lengthx = countedx.len();
    //let mut priorx = countedx[0];
    let mut runningtotalx = 0;
    let mut outgoingx: Vec<usize> = vec![];

    for each in 0usize..uniques_lengthx {
        runningtotalx += countedx[each];
        let wholesx: usize = runningtotalx.saturating_sub( countedx[each] );
        let halvesx:usize = countedx[each].saturating_sub(1);
        let doubledrankx = 2*wholesx + halvesx;
        for _every in 0..countedx[each] {
              outgoingx.push(doubledrankx.clone());
        }
      //  priorx = runningtotalx;
    }

    let mut originx:Vec<usize> = vec![0usize ; datax.len()];
    for each in 0..(outgoingx.len())  {
        originx[arrangementx[each]] =  outgoingx[each]
    }
  //then rank y data

    let arrangementy:Vec<usize> = rank(&datay);
    let repattery: Vec<P> = arrangementy.iter().map(|v| datay[*v]).collect();
    let countedy: Vec<usize> = count_sorted_duplicates_or_uniques(&repattery);
    let uniques_lengthy = countedy.len();
    //let mut priory = countedy[0];
    let mut runningtotaly = 0;
    let mut outgoingy: Vec<usize> = vec![];

    for each in 0usize..uniques_lengthy {
        runningtotaly += countedy[each];
        let wholesy: usize = runningtotaly.saturating_sub( countedy[each] );
        let halvesy:usize = countedy[each].saturating_sub(1);
        let doubledranky = 2*wholesy + halvesy;
        for _every in 0..countedy[each] {
              outgoingy.push(doubledranky.clone());
        }
       // priory = runningtotaly;
    }

    let mut originy:Vec<usize> = vec![0usize ; datay.len()];
    for each in 0..(outgoingy.len())  {
        originy[arrangementy[each]] =  outgoingy[each]
    }

    // and now on to the tedious rank of xy data where being greater in X & Y counts for 4 points, being greater on 1 axis and matched counts for 2 points, and matching each axis counts for 1 point of rank 
   let mut originxy:Vec<usize> = vec![];

   for j in 0..n_element { //wish there was a less insanity laden path than n^2 - but majority of paried x&y data comparisons will not be be deftly organized, low noise and strongly correlated.
     let v:usize =   (0..n_element).map(|i|  2*((originx[i]<originx[j] && originy[i]<originy[j] )as usize) + 
                                                          2*((originx[i]<=originx[j] && originy[i]<=originy[j]) as usize) -
                                                          ((originx[i]==originy[j] && originy[i]==originy[j]) as usize)   
                    ).collect::<Vec<usize>>().iter().sum();
                    let v = v-1; //X and Y ranks do not self compare but XY ranks do in a way described in Hoeffding's 1957 paper, unfortunately in wording that is unclear to me...  Should the XY ith point SELF comparison (i==i) give a contribution of zero, one quarter, or four quarters?     
    originxy.push(v);            //
   }   
 
    let d_subone_times_four_list:Vec<i128> = (0..n_element).map(|i|  ((originxy[i] as i128)-4) * (((originxy[i] as i128)) -8  )).collect();  //note that originxy[i] = Q_sub_i - 1 (bivariate rank minus 1 for each i)
    let d_subone_times_four:i128 = d_subone_times_four_list.iter().sum();
    let d_subtwo_times_two_list:Vec<i128>= (0..n_element).map(|i| ((originx[i] as i128) -2) * ((originx[i] as i128)-4) * ((originy[i] as i128)-2)*((originy[i] as i128)-4)).collect() ;
    let d_subtwo_times_two:i128 = d_subtwo_times_two_list.iter().sum();
    let d_subthree_times_two_list:Vec<i128> = (0..n_element).map(|i| ((originx[i] as i128)-4i128 )*((originy[i] as i128)-4i128)*((originxy[i]as i128 )-4i128)  ).collect();
    let d_subthree_time_two: i128 = d_subthree_times_two_list.iter().sum();
    let n = n_element as i128;
    let pockmark:i128= - n*(n-1)*(n-2)*(n-3)*(n-4);
    let h_first:i128 = (( ((n-2)*(n-3)) * d_subone_times_four) as i128)/(4);  //n is greater than 4 and therefore 30*(n-2)*(n-3) is always positive and evenly divible by four.  
    let h_second:i128 = ( d_subtwo_times_two / 2) as i128;
    let h_third:i128 = ((n-2)*d_subthree_time_two) as i128; //note (2n-4) becomes n-2 to account for multiply by two
    
   let  mut hoeffding_partial_numerator:i128 = h_first + h_second - h_third;

hoeffding_partial_numerator // linear porportianilty to actual statistic, but not normalized  - normalization calculation is computationally intensive as just getting the numerator
    //and yet  

}
fn rank_on_climbing_sorted_data_value_doubled<T: PartialOrd + Clone + Copy + Debug >(data: &Vec<T>) -> Vec<usize> {
    let arrangement: Vec<usize> = rank(&data);
    let repatter: Vec<T> = arrangement.iter().map(|v| data[*v]).collect();
    dbg!(&arrangement);
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
    let data: Vec<&str> = vec!["a","a","c","b","b","b","f","d","d","e","e","e"];
    let data2: Vec<&str> = vec!["c","b","b","b","f","d","d","e","e","e","f","a"];
    let temp: Vec<usize> = rank(&data);
    let repatter: Vec<&str> = temp.iter().map(|v| data[*v].clone()).collect();

    println!(
        "data: {:?} temp: {:?} restruct:{:?}",
        &data, &temp, &repatter
    );

    println!("testing:  {:?}", &repatter);

    let countr = count_sorted_duplicates_or_uniques(&repatter);

    println!("counted: {:?}", &countr);

    println!("2xranks: {:?}", rank_on_climbing_sorted_data_value_doubled(&data));
    println!("   data: {:?}",&data);
   
    let hoeff = hoeffding_health(&data, &data2);
  
println!("   data2 {:?}",&data2);
    println!("filter data:{:?}",hoeff);
println!();

for rep in 5..6 {
let mut longdataa: Vec<i128> = vec![];
let mut longdatab: Vec<i128> = vec![];

for each in 0..rep {
    let a = 0;//rng.gen();
  // let a:f64 = rng.gen();
  let b = 1;
              //  0.0f64;
    //let b: = ("a","b","c"));
    //let b = 3. *a + b;
    longdataa.push(a);
longdatab.push(b);
}
//let now1 = std::time::Instant::now();
let d = hoeffding_health(&longdataa, &longdatab);//let hoeff = hoeffding_dependence_coefficient(&longdataa, &longdatab);
print!(", {}",d);
if rep%10==0 {println!();}
//let now2= std::time::Instant::now(); let timekeeper = now2-now1;
}

println!("i 128 max {}",i128::MAX);

}
