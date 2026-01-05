/* LOST BAGAGGE DISTRIBUTION PROBLEM - MIP Optimization
The used mathematical method is not the same than the suggested in the book, but it leads to the same results in a more optimized process that searches for both solutions simultaneously. The compact notation is included before each extended formulation.
This is the same model as the one used in Excel, with the compact notation being the same as the presented in the spreadsheet. Here it is possible to exclude all i=j diagonal.
There is a total of 421 variables of which 210 are binary and 253 functional constraints (including linking and subtour elimination).

/* Objective function */
    // min: M SUM[i=1 to 15](x_1j) + T
    min: 100V + T;
    V = x1_2 + x1_3 + x1_4 + x1_5 + x1_6 + x1_7 + x1_8 + x1_9 + x1_10 + x1_11 + x1_12 + x1_13 + x1_14 + x1_15;
    
    /* Since Heathrow is the partitioning point, this formulae focuses on finding how many times it is necessary to start a new network to visit all the places within the time constraint. Then, a slack variable analogy is introduced to constraint the
    maximum time to visit all the locations for each van, so minimizing this leads to solving the Minmax problem of longest route. Minimizing the whole problem requires a big M to prioritize finding the lowest number of vans needed.
    In this program, V is just a helper variable to track the number of vans needed.
*/

/* Functional Constraints */
// (1) Every destination must be visited once (except Heathrow):
// Columns:
    // SUM[i=1 to 15](x_ij) = 1   for j = 2,...,15
    x2_1 + x2_3 + x2_4 + x2_5 + x2_6 + x2_7 + x2_8 + x2_9 + x2_10 + x2_11 + x2_12 + x2_13 + x2_14 + x2_15 = 1;
    x3_1 + x3_2 + x3_4 + x3_5 + x3_6 + x3_7 + x3_8 + x3_9 + x3_10 + x3_11 + x3_12 + x3_13 + x3_14 + x3_15 = 1;
    x4_1 + x4_2 + x4_3 + x4_5 + x4_6 + x4_7 + x4_8 + x4_9 + x4_10 + x4_11 + x4_12 + x4_13 + x4_14 + x4_15 = 1;
    x5_1 + x5_2 + x5_3 + x5_4 + x5_6 + x5_7 + x5_8 + x5_9 + x5_10 + x5_11 + x5_12 + x5_13 + x5_14 + x5_15 = 1;
    x6_1 + x6_2 + x6_3 + x6_4 + x6_5 + x6_7 + x6_8 + x6_9 + x6_10 + x6_11 + x6_12 + x6_13 + x6_14 + x6_15 = 1;
    x7_1 + x7_2 + x7_3 + x7_4 + x7_5 + x7_6 + x7_8 + x7_9 + x7_10 + x7_11 + x7_12 + x7_13 + x7_14 + x7_15 = 1;
    x8_1 + x8_2 + x8_3 + x8_4 + x8_5 + x8_6 + x8_7 + x8_9 + x8_10 + x8_11 + x8_12 + x8_13 + x8_14 + x8_15 = 1;
    x9_1 + x9_2 + x9_3 + x9_4 + x9_5 + x9_6 + x9_7 + x9_8 + x9_10 + x9_11 + x9_12 + x9_13 + x9_14 + x9_15 = 1;
    x10_1 + x10_2 + x10_3 + x10_4 + x10_5 + x10_6 + x10_7 + x10_8 + x10_9 + x10_11 + x10_12 + x10_13 + x10_14 + x10_15 = 1;
    x11_1 + x11_2 + x11_3 + x11_4 + x11_5 + x11_6 + x11_7 + x11_8 + x11_9 + x11_10 + x11_12 + x11_13 + x11_14 + x11_15 = 1;
    x12_1 + x12_2 + x12_3 + x12_4 + x12_5 + x12_6 + x12_7 + x12_8 + x12_9 + x12_10 + x12_11 + x12_13 + x12_14 + x12_15 = 1;
    x13_1 + x13_2 + x13_3 + x13_4 + x13_5 + x13_6 + x13_7 + x13_8 + x13_9 + x13_10 + x13_11 + x13_12 + x13_14 + x13_15 = 1;
    x14_1 + x14_2 + x14_3 + x14_4 + x14_5 + x14_6 + x14_7 + x14_8 + x14_9 + x14_10 + x14_11 + x14_12 + x14_13 + x14_15 = 1;
    x15_1 + x15_2 + x15_3 + x15_4 + x15_5 + x15_6 + x15_7 + x15_8 + x15_9 + x15_10 + x15_11 + x15_12 + x15_13 + x15_14 = 1;
    
// Rows:
    // SUM[j=1 to 15](x_ij) = 1   for i = 2,...,15
    x1_2 + x3_2 + x4_2 + x5_2 + x6_2 + x7_2 + x8_2 + x9_2 + x10_2 + x11_2 + x12_2 + x13_2 + x14_2 + x15_2 = 1;
    x1_3 + x2_3 + x4_3 + x5_3 + x6_3 + x7_3 + x8_3 + x9_3 + x10_3 + x11_3 + x12_3 + x13_3 + x14_3 + x15_3 = 1;
    x1_4 + x2_4 + x3_4 +  x5_4 + x6_4 + x7_4 + x8_4 + x9_4 + x10_4 + x11_4 + x12_4 + x13_4 + x14_4 + x15_4 = 1;
    x1_5 + x2_5 + x3_5 + x4_5 + x6_5 + x7_5 + x8_5 + x9_5 + x10_5 + x11_5 + x12_5 + x13_5 + x14_5 + x15_5 = 1;
    x1_6 + x2_6 + x3_6 + x4_6 + x5_6 + x7_6 + x8_6 + x9_6 + x10_6 + x11_6 + x12_6 + x13_6 + x14_6 + x15_6 = 1;
    x1_7 + x2_7 + x3_7 + x4_7 + x5_7 + x6_7 + x8_7 + x9_7 + x10_7 + x11_7 + x12_7 + x13_7 + x14_7 + x15_7 = 1;
    x1_8 + x2_8 + x3_8 + x4_8 + x5_8 + x6_8 + x7_8 + x9_8 + x10_8 + x11_8 + x12_8 + x13_8 + x14_8 + x15_8 = 1;
    x1_9 + x2_9 + x3_9 + x4_9 + x5_9 + x6_9 + x7_9 + x8_9 + x10_9 + x11_9 + x12_9 + x13_9 + x14_9 + x15_9 = 1;
    x1_10 + x2_10 + x3_10 + x4_10 + x5_10 + x6_10 + x7_10 + x8_10 + x9_10 + x11_10 + x12_10 + x13_10 + x14_10 + x15_10 = 1;
    x1_11 + x2_11 + x3_11 + x4_11 + x5_11 + x6_11 + x7_11 + x8_11 + x9_11 + x10_11 + x12_11 + x13_11 + x14_11 + x15_11 = 1;
    x1_12 + x2_12 + x3_12 + x4_12 + x5_12 + x6_12 + x7_12 + x8_12 + x9_12 + x10_12 + x11_12 + x13_12 + x14_12 + x15_12 = 1;
    x1_13 + x2_13 + x3_13 + x4_13 + x5_13 + x6_13 + x7_13 + x8_13 + x9_13 + x10_13 + x11_13 + x12_13 + x14_13 + x15_13 = 1;
    x1_14 + x2_14 + x3_14 + x4_14 + x5_14 + x6_14 + x7_14 + x8_14 + x9_14 + x10_14 + x11_14 + x12_14 + x13_14 + x15_14 = 1;
    x1_15 + x2_15 + x3_15 + x4_15 + x5_15 + x6_15 + x7_15 + x8_15 + x9_15 + x10_15 + x11_15 + x12_15 + x13_15 + x14_15 = 1;
    
    /* Using square matrix allows to control for visiting all the cities with one link in and one link out, which is the same logic that the book uses.
*/

// (2) Limit of abviable vans:
    // SUM[j=1 to 15](x_1j) <= 6
    x1_2 + x1_3 + x1_4 + x1_5 + x1_6 + x1_7 + x1_8 + x1_9 + x1_10 + x1_11 + x1_12 + x1_13 + x1_14 + x1_15 <= 6;
    
    /* Since Heathrow is not constraint to more than 1 visit as the book states, it is possible to merge this in one matrix and constraint the number of links out representing the number of vans.
*/
    
// (3) 120 minutes for each van to complete its tour (no return to Heathrow needed):
    // f_ij <= 120x_ij
    f1_2 <= 120x1_2; f1_3 <= 120x1_3; f1_4 <= 120x1_4; f1_5 <= 120x1_5; f1_6 <= 120x1_6; f1_7 <= 120x1_7; f1_8 <= 120x1_8; f1_9 <= 120x1_9; f1_10 <= 120x1_10; f1_11 <= 120x1_11; f1_12 <= 120x1_12; f1_13 <= 120x1_13; f1_14 <= 120x1_14; f1_15 <= 120x1_15;
    f2_1 <= 120x2_1; f2_3 <= 120x2_3; f2_4 <= 120x2_4; f2_5 <= 120x2_5; f2_6 <= 120x2_6; f2_7 <= 120x2_7; f2_8 <= 120x2_8; f2_9 <= 120x2_9; f2_10 <= 120x2_10; f2_11 <= 120x2_11; f2_12 <= 120x2_12; f2_13 <= 120x2_13; f2_14 <= 120x2_14; f2_15 <= 120x2_15;
    f3_1 <= 120x3_1; f3_2 <= 120x3_2; f3_4 <= 120x3_4; f3_5 <= 120x3_5; f3_6 <= 120x3_6; f3_7 <= 120x3_7; f3_8 <= 120x3_8; f3_9 <= 120x3_9; f3_10 <= 120x3_10; f3_11 <= 120x3_11; f3_12 <= 120x3_12; f3_13 <= 120x3_13; f3_14 <= 120x3_14; f3_15 <= 120x3_15;
    f4_1 <= 120x4_1; f4_2 <= 120x4_2; f4_3 <= 120x4_3; f4_5 <= 120x4_5; f4_6 <= 120x4_6; f4_7 <= 120x4_7; f4_8 <= 120x4_8; f4_9 <= 120x4_9; f4_10 <= 120x4_10; f4_11 <= 120x4_11; f4_12 <= 120x4_12; f4_13 <= 120x4_13; f4_14 <= 120x4_14; f4_15 <= 120x4_15;
    f5_1 <= 120x5_1; f5_2 <= 120x5_2; f5_3 <= 120x5_3; f5_4 <= 120x5_4; f5_6 <= 120x5_6; f5_7 <= 120x5_7; f5_8 <= 120x5_8; f5_9 <= 120x5_9; f5_10 <= 120x5_10; f5_11 <= 120x5_11; f5_12 <= 120x5_12; f5_13 <= 120x5_13; f5_14 <= 120x5_14; f5_15 <= 120x5_15;
    f6_1 <= 120x6_1; f6_2 <= 120x6_2; f6_3 <= 120x6_3; f6_4 <= 120x6_4; f6_5 <= 120x6_5; f6_7 <= 120x6_7; f6_8 <= 120x6_8; f6_9 <= 120x6_9; f6_10 <= 120x6_10; f6_11 <= 120x6_11; f6_12 <= 120x6_12; f6_13 <= 120x6_13; f6_14 <= 120x6_14; f6_15 <= 120x6_15;
    f7_1 <= 120x7_1; f7_2 <= 120x7_2; f7_3 <= 120x7_3; f7_4 <= 120x7_4; f7_5 <= 120x7_5; f7_6 <= 120x7_6; f7_8 <= 120x7_8; f7_9 <= 120x7_9; f7_10 <= 120x7_10; f7_11 <= 120x7_11; f7_12 <= 120x7_12; f7_13 <= 120x7_13; f7_14 <= 120x7_14; f7_15 <= 120x7_15;
    f8_1 <= 120x8_1; f8_2 <= 120x8_2; f8_3 <= 120x8_3; f8_4 <= 120x8_4; f8_5 <= 120x8_5; f8_6 <= 120x8_6; f8_7 <= 120x8_7; f8_9 <= 120x8_9; f8_10 <= 120x8_10; f8_11 <= 120x8_11; f8_12 <= 120x8_12; f8_13 <= 120x8_13; f8_14 <= 120x8_14; f8_15 <= 120x8_15;
    f9_1 <= 120x9_1; f9_2 <= 120x9_2; f9_3 <= 120x9_3; f9_4 <= 120x9_4; f9_5 <= 120x9_5; f9_6 <= 120x9_6; f9_7 <= 120x9_7; f9_8 <= 120x9_8; f9_10 <= 120x9_10; f9_11 <= 120x9_11; f9_12 <= 120x9_12; f9_13 <= 120x9_13; f9_14 <= 120x9_14; f9_15 <= 120x9_15;
    f10_1 <= 120x10_1; f10_2 <= 120x10_2; f10_3 <= 120x10_3; f10_4 <= 120x10_4; f10_5 <= 120x10_5; f10_6 <= 120x10_6; f10_7 <= 120x10_7; f10_8 <= 120x10_8; f10_9 <= 120x10_9; f10_11 <= 120x10_11; f10_12 <= 120x10_12; f10_13 <= 120x10_13; f10_14 <= 120x10_14; f10_15 <= 120x10_15;
    f11_1 <= 120x11_1; f11_2 <= 120x11_2; f11_3 <= 120x11_3; f11_4 <= 120x11_4; f11_5 <= 120x11_5; f11_6 <= 120x11_6; f11_7 <= 120x11_7; f11_8 <= 120x11_8; f11_9 <= 120x11_9; f11_10 <= 120x11_10; f11_12 <= 120x11_12; f11_13 <= 120x11_13; f11_14 <= 120x11_14; f11_15 <= 120x11_15;
    f12_1 <= 120x12_1; f12_2 <= 120x12_2; f12_3 <= 120x12_3; f12_4 <= 120x12_4; f12_5 <= 120x12_5; f12_6 <= 120x12_6; f12_7 <= 120x12_7; f12_8 <= 120x12_8; f12_9 <= 120x12_9; f12_10 <= 120x12_10; f12_11 <= 120x12_11; f12_13 <= 120x12_13; f12_14 <= 120x12_14; f12_15 <= 120x12_15;
    f13_1 <= 120x13_1; f13_2 <= 120x13_2; f13_3 <= 120x13_3; f13_4 <= 120x13_4; f13_5 <= 120x13_5; f13_6 <= 120x13_6; f13_7 <= 120x13_7; f13_8 <= 120x13_8; f13_9 <= 120x13_9; f13_10 <= 120x13_10; f13_11 <= 120x13_11; f13_12 <= 120x13_12; f13_14 <= 120x13_14; f13_15 <= 120x13_15;
    f14_1 <= 120x14_1; f14_2 <= 120x14_2; f14_3 <= 120x14_3; f14_4 <= 120x14_4; f14_5 <= 120x14_5; f14_6 <= 120x14_6; f14_7 <= 120x14_7; f14_8 <= 120x14_8; f14_9 <= 120x14_9; f14_10 <= 120x14_10; f14_11 <= 120x14_11; f14_12 <= 120x14_12; f14_13 <= 120x14_13; f14_15 <= 120x14_15;
    f15_1 <= 120x15_1; f15_2 <= 120x15_2; f15_3 <= 120x15_3; f15_4 <= 120x15_4; f15_5 <= 120x15_5; f15_6 <= 120x15_6; f15_7 <= 120x15_7; f15_8 <= 120x15_8; f15_9 <= 120x15_9; f15_10 <= 120x15_10; f15_11 <= 120x15_11; f15_12 <= 120x15_12; f15_13 <= 120x15_13; f15_14 <= 120x15_14;
    
    /* The linking flow logic takes the budget (time) and substracts the cost of the inflow and the outflow of each node. Balancing that, means that any variable that is part of the network must be less than 120, which is the total time for each network.
    It is important not to constraint the whole matrix to a fixed number, but constraint iif the variable is part of the tour because that sets the budget and ignores the nodes <=0.
*/

// (4) Minimize the time abviable for each tour:
    // f_1j <= T   for j = 2,...,15
    f1_2 <= T;
    f1_3 <= T;
    f1_4 <= T;
    f1_5 <= T;
    f1_6 <= T;
    f1_7 <= T;
    f1_8 <= T;
    f1_9 <= T;
    f1_10 <= T;
    f1_11 <= T;
    f1_12 <= T;
    f1_13 <= T;
    f1_14 <= T;
    f1_15 <= T;
    
    /* Since the structure of the problem starts by setting the required time to visit all cities for each van, constraining each element of the first row of the flow matrix with a slack variable analogy (suggested in the book) forces to minimize the
    time in any route. This is part of the objective function, but was included in the constraints part since it pairs with the logic of time limit. Nevertheless, minimizing the the time limit itself leads to a non-linear problem.
    
/* Linking and Subtour Elimination Constraints */
// (5) Flow balance with Time Values from Matrix:
    // SUM[i=1 to 15](f_ij) - SUM[i=1 to 15](f_ji) = SUM[i=1 to 15](c_ij x_ij)   for j = 2,...,15
    f1_2 + f3_2 + f4_2 + f5_2 + f6_2 + f7_2 + f8_2 + f9_2 + f10_2 + f11_2 + f12_2 + f13_2 + f14_2 + f15_2 -
    f2_1 - f2_3 - f2_4 - f2_5 - f2_6 - f2_7 - f2_8 - f2_9 - f2_10 - f2_11 - f2_12 - f2_13 - f2_14 - f2_15 =
    20x1_2 + 15x3_2 + 35x4_2 + 60x5_2 + 55x6_2 + 57x7_2 + 85x8_2 + 90x9_2 + 25x10_2 + 35x11_2 + 30x12_2 + 37x13_2 + 20x14_2 + 40x15_2;

    f1_3 + f2_3 + f4_3 + f5_3 + f6_3 + f7_3 + f8_3 + f9_3 + f10_3 + f11_3 + f12_3 + f13_3 + f14_3 + f15_3 -
    f3_1 - f3_2 - f3_4 - f3_5 - f3_6 - f3_7 - f3_8 - f3_9 - f3_10 - f3_11 - f3_12 - f3_13 - f3_14 - f3_15 =
    25x1_3 + 15x2_3 + 30x4_3 + 50x5_3 + 70x6_3 + 55x7_3 + 50x8_3 + 65x9_3 + 10x10_3 + 25x11_3 + 15x12_3 + 24x13_3 + 20x14_3 + 90x15_3;

    f1_4 + f2_4 + f3_4 + f5_4 + f6_4 + f7_4 + f8_4 + f9_4 + f10_4 + f11_4 + f12_4 + f13_4 + f14_4 + f15_4 -
    f4_1 - f4_2 - f4_3 - f4_5 - f4_6 - f4_7 - f4_8 - f4_9 - f4_10 - f4_11 - f4_12 - f4_13 - f4_14 - f4_15 =
    35x1_4 + 35x2_4 + 30x3_4 + 45x5_4 + 60x6_4 + 53x7_4 + 55x8_4 + 47x9_4 + 12x10_4 + 22x11_4 + 20x12_4 + 12x13_4 + 10x14_4 + 21x15_4;

    f1_5 + f2_5 + f3_5 + f4_5 + f6_5 + f7_5 + f8_5 + f9_5 + f10_5 + f11_5 + f12_5 + f13_5 + f14_5 + f15_5 -
    f5_1 - f5_2 - f5_3 - f5_4 - f5_6 - f5_7 - f5_8 - f5_9 - f5_10 - f5_11 - f5_12 - f5_13 - f5_14 - f5_15 =
    65x1_5 + 60x2_5 + 50x3_5 + 45x4_5 + 46x6_5 + 15x7_5 + 45x8_5 + 75x9_5 + 25x10_5 + 11x11_5 + 19x12_5 + 15x13_5 + 25x14_5 + 25x15_5;

    f1_6 + f2_6 + f3_6 + f4_6 + f5_6 + f7_6 + f8_6 + f9_6 + f10_6 + f11_6 + f12_6 + f13_6 + f14_6 + f15_6 -
    f6_1 - f6_2 - f6_3 - f6_4 - f6_5 - f6_7 - f6_8 - f6_9 - f6_10 - f6_11 - f6_12 - f6_13 - f6_14 - f6_15 =
    90x1_6 + 55x2_6 + 70x3_6 + 60x4_6 + 46x5_6 + 15x7_6 + 15x8_6 + 25x9_6 + 45x10_6 + 65x11_6 + 53x12_6 + 43x13_6 + 63x14_6 + 70x15_6;

    f1_7 + f2_7 + f3_7 + f4_7 + f5_7 + f6_7 + f8_7 + f9_7 + f10_7 + f11_7 + f12_7 + f13_7 + f14_7 + f15_7 -
    f7_1 - f7_2 - f7_3 - f7_4 - f7_5 - f7_6 - f7_8 - f7_9 - f7_10 - f7_11 - f7_12 - f7_13 - f7_14 - f7_15 =
    85x1_7 + 57x2_7 + 55x3_7 + 53 x4_7 + 15x5_7 + 15x6_7 + 17x8_7 + 25x9_7 + 41x10_7 + 25x11_7 + 33x12_7 + 27x13_7 + 45x14_7 + 30x15_7;

    f1_8 + f2_8 + f3_8 + f4_8 + f5_8 + f6_8 + f7_8 + f9_8 + f10_8 + f11_8 + f12_8 + f13_8 + f14_8 + f15_8 -
    f8_1 - f8_2 - f8_3 - f8_4 - f8_5 - f8_6 - f8_7 - f8_9 - f8_10 - f8_11 - f8_12 - f8_13 - f8_14 - f8_15 =
    80x1_8 + 85x2_8 + 50x3_8 + 55x4_8 + 45x5_8 + 15x6_8 + 17x7_8 + 25x9_8 + 40x10_8 + 34x11_8 + 32x12_8 + 20x13_8 + 30x14_8 + 10x15_8;

    f1_9 + f2_9 + f3_9 + f4_9 + f5_9 + f6_9 + f7_9 + f8_9 + f10_9 + f11_9 + f12_9 + f13_9 + f14_9 + f15_9 -
    f9_1 - f9_2 - f9_3 - f9_4 - f9_5 - f9_6 - f9_7 - f9_8 - f9_10 - f9_11 - f9_12 - f9_13 - f9_14 - f9_15 =
    86x1_9 + 90x2_9 + 65x3_9 + 47x4_9 + 75x5_9 + 25x6_9 + 25x7_9 + 25x8_9 + 65x10_9 + 70x11_9 + 72x12_9 + 61x13_9 + 45x14_9 + 13x15_9;

    f1_10 + f2_10 + f3_10 + f4_10 + f5_10 + f6_10 + f7_10 + f8_10 + f9_10 + f11_10 + f12_10 + f13_10 + f14_10 + f15_10 -
    f10_1 - f10_2 - f10_3 - f10_4 - f10_5 - f10_6 - f10_7 - f10_8 - f10_9 - f10_11 - f10_12 - f10_13 - f10_14 - f10_15 =
    25x1_10 + 25x2_10 + 10x3_10 + 12x4_10 + 25x5_10 + 45x6_10 + 41x7_10 + 40x8_10 + 65x9_10 + 20x11_10 + 8x12_10 + 7x13_10 + 15x14_10 + 25x15_10;

    f1_11 + f2_11 + f3_11 + f4_11 + f5_11 + f6_11 + f7_11 + f8_11 + f9_11 + f10_11 + f12_11 + f13_11 + f14_11 + f15_11 -
    f11_1 - f11_2 - f11_3 - f11_4 - f11_5 - f11_6 - f11_7 - f11_8 - f11_9 - f11_10 - f11_12 - f11_13 - f11_14 - f11_15 =
    35x1_11 + 35x2_11 + 25x3_11 + 22x4_11 + 11x5_11 + 65x6_11 + 25x7_11 + 34x8_11 + 70x9_11 + 20x10_11 + 5x12_11 + 12x13_11 + 45x14_11 + 56x15_11;

    f1_12 + f2_12 + f3_12 + f4_12 + f5_12 + f6_12 + f7_12 + f8_12 + f9_12 + f10_12 + f11_12 + f13_12 + f14_12 + f15_12 -
    f12_1 - f12_2 - f12_3 - f12_4 - f12_5 - f12_6 - f12_7 - f12_8 - f12_9 - f12_10 - f12_11 - f12_13 - f12_14 - f12_15 =
    20x1_12 + 30x2_12 + 15x3_12 + 20x4_12 + 19x5_12 + 53x6_12 + 33x7_12 + 32x8_12 + 72x9_12 + 8x10_12 + 5x11_12 + 14x13_12 + 34x14_12 + 40x15_12;

    f1_13 + f2_13 + f3_13 + f4_13 + f5_13 + f6_13 + f7_13 + f8_13 + f9_13 + f10_13 + f11_13 + f12_13 + f14_13 + f15_13 -
    f13_1 - f13_2 - f13_3 - f13_4 - f13_5 - f13_6 - f13_7 - f13_8 - f13_9 - f13_10 - f13_11 - f13_12 - f13_14 - f13_15 =
    44x1_13 + 37x2_13 + 24x3_13 + 12x4_13 + 15x5_13 + 43x6_13 + 27x7_13 + 20x8_13 + 61x9_13 + 7x10_13 + 12x11_13 + 14x12_13 + 30x14_13 + 40x15_13;

    f1_14 + f2_14 + f3_14 + f4_14 + f5_14 + f6_14 + f7_14 + f8_14 + f9_14 + f10_14 + f11_14 + f12_14 + f13_14 + f15_14 -
    f14_1 - f14_2 - f14_3 - f14_4 - f14_5 - f14_6 - f14_7 - f14_8 - f14_9 - f14_10 - f14_11 - f14_12 - f14_13 - f14_15 =
    35x1_14 + 20x2_14 + 20x3_14 + 10x4_14 + 25x5_14 + 63x6_14 + 45x7_14 + 30x8_14 + 45x9_14 + 15x10_14 + 45x11_14 + 34x12_14 + 30x13_14 + 27x15_14;

    f1_15 + f2_15 + f3_15 + f4_15 + f5_15 + f6_15 + f7_15 + f8_15 + f9_15 + f10_15 + f11_15 + f12_15 + f13_15 + f14_15 -
    f15_1 - f15_2 - f15_3 - f15_4 - f15_5 - f15_6 - f15_7 - f15_8 - f15_9 - f15_10 - f15_11 - f15_12 - f15_13 - f15_14 =
    82x1_15 + 40x2_15 + 90x3_15 + 21x4_15 + 25x5_15 + 70x6_15 + 30x7_15 + 10x8_15 + 13x9_15 + 25x10_15 + 65x11_15 + 56x12_15 + 40x13_15 + 27x14_15;
    
    /* To keep the one link in - one link out logic it is only necessary to set the cost of returning to Heathrow as 0 as the book suggested. Then, balancing the outflow cost to match the used time to travel to other city will be mathematically
    consistent with consuming from the total required time to finish a loop. In the end, this works hand by hand with the linking matrix, so that if a variable is constrained to 0 it is not taken into consideration for the route optimization. Finally,
    the problem being solved is a matter of how many loops are necessary to visit all the cities where Heathrow is the starting and the end point before starting a new network. This logic minimizes the number of required variables to solve the problem
    in comparison to the book solution, while being accurate for any similar case scenario.
*/

/* Binding Constraints */
bin x1_2, x1_3, x1_4, x1_5, x1_6, x1_7, x1_8, x1_9, x1_10, x1_11, x1_12, x1_13, x1_14, x1_15;
bin x2_1, x2_3, x2_4, x2_5, x2_6, x2_7, x2_8, x2_9, x2_10, x2_11, x2_12, x2_13, x2_14, x2_15;
bin x3_1, x3_2, x3_4, x3_5, x3_6, x3_7, x3_8, x3_9, x3_10, x3_11, x3_12, x3_13, x3_14, x3_15;
bin x4_1, x4_2, x4_3, x4_5, x4_6, x4_7, x4_8, x4_9, x4_10, x4_11, x4_12, x4_13, x4_14, x4_15;
bin x5_1, x5_2, x5_3, x5_4, x5_6, x5_7, x5_8, x5_9, x5_10, x5_11, x5_12, x5_13, x5_14, x5_15;
bin x6_1, x6_2, x6_3, x6_4, x6_5, x6_7, x6_8, x6_9, x6_10, x6_11, x6_12, x6_13, x6_14, x6_15;
bin x7_1, x7_2, x7_3, x7_4, x7_5, x7_6, x7_8, x7_9, x7_10, x7_11, x7_12, x7_13, x7_14, x7_15;
bin x8_1, x8_2, x8_3, x8_4, x8_5, x8_6, x8_7, x8_9, x8_10, x8_11, x8_12, x8_13, x8_14, x8_15;
bin x9_1, x9_2, x9_3, x9_4, x9_5, x9_6, x9_7, x9_8, x9_10, x9_11, x9_12, x9_13, x9_14, x9_15;
bin x10_1, x10_2, x10_3, x10_4, x10_5, x10_6, x10_7, x10_8, x10_9, x10_11, x10_12, x10_13, x10_14, x10_15;
bin x11_1, x11_2, x11_3, x11_4, x11_5, x11_6, x11_7, x11_8, x11_9, x11_10, x11_12, x11_13, x11_14, x11_15;
bin x12_1, x12_2, x12_3, x12_4, x12_5, x12_6, x12_7, x12_8, x12_9, x12_10, x12_11, x12_13, x12_14, x12_15;
bin x13_1, x13_2, x13_3, x13_4, x13_5, x13_6, x13_7, x13_8, x13_9, x13_10, x13_11, x13_12, x13_14, x13_15;
bin x14_1, x14_2, x14_3, x14_4, x14_5, x14_6, x14_7, x14_8, x14_9, x14_10, x14_11, x14_12, x14_13, x14_15;
bin x15_1, x15_2, x15_3, x15_4, x15_5, x15_6, x15_7, x15_8, x15_9, x15_10, x15_11, x15_12, x15_13, x15_14;

/* In this solver it is possible to eliminate any i=j, while in the Excel was necessary to set an infeasible cost for the diagonal to be ignored. All the decision variables are binary and the flow variables are continuous.
*/
