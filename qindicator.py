"""
Quality indicators for preference-based evolutionary multi-objective optimization using a reference point
"""
import os
import numpy as np
import pylab as plt
from scipy.stats import rankdata
import pygmo as pg
from scipy import spatial
from pymoo.factory import get_problem, get_performance_indicator
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def asf(point, ref_point, weight):
    scalar_value = np.max((point - ref_point) / weight)    
    return scalar_value

def asf1(point, ref_point, weight):
    scalar_value = np.max(weight * (point - ref_point))
    return scalar_value    

def tchebycheff(weight_vec, ideal_point, point):
    tch_val = -np.inf
    for w, z, f in zip(weight_vec, ideal_point, point):
        tch_val = max(tch_val, w * abs(f - z))
    return tch_val

def igd(ref_point_set, point_set):
    """   
    Calculate the IGD value of given point sets.

    This function is derived from "quality_indicator.py" in jMetalPy (https://github.com/jMetal/jMetalPy/blob/master/jmetal/core/quality_indicator.py)

    Parameters
    ----------
    ref_point_set: 2-d float numpy ndarray
        A reference point set for the IGD calculation
    point_set: 2-d float numpy ndarray
        A point set
    """        
    distances = spatial.distance.cdist(ref_point_set, point_set, metric='euclidean')
    return np.mean(np.min(distances, axis=1))

def igdplus(ref_point_set, point_set):
    """   
    Calculate the IGD^+ value of given point sets.

    Parameters
    ----------
    ref_point_set: 2-d float numpy ndarray
        A reference point set for the IGD calculation
    point_set: 2-d float numpy ndarray
        A point set
    """        
    sum_dist = 0
    for rp in ref_point_set:                    
        min_dist = np.inf
        for p in point_set:        
            tmp = 0
            for x, r in zip(p, rp):
                tmp += (max(x - r, 0))**2                        
            tmp_dist = np.sqrt(tmp)
            min_dist = min(min_dist, tmp_dist)                
        sum_dist += min_dist
    return sum_dist / len(ref_point_set)

def r2(weight_vec_set, ideal_point, point_set):  
    sum_tch = 0
    for w in weight_vec_set:
        min_tch = np.inf
        for p in point_set:            
            tch = tchebycheff(w, ideal_point, p)
            min_tch = min(min_tch, tch)
        sum_tch += min_tch
    return sum_tch / len(weight_vec_set)

def minimum_asf(pset_file_path, qi_file_path, ref_point, weight_vec):
    """   
    Calculate the minimum ASF value in a given point set.

    [Reference] L. Thiele, K. Miettinen, P. J. Korhonen, and J. M. Luque, "A preference-based evolutionary algorithm for multi-objective optimization," Evol. Comput., vol. 17, no. 3, pp. 411-436, 2009.

    Parameters
    ----------
    pset_file_path: file path         
        A file path to load the point set 
    qi_file_path: file path         
        A file path to save the quality indicator value
    ref_point: 1-d float numpy ndarray
        A reference point (an aspiration vector)
    weight_vec: 1-d float numpy ndarray
        A vector to determine a relative importance of each objective function
    """    
    point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)    
    asf_list = []
    for p in point_set:
        asf_list.append(asf(p, ref_point, weight_vec))
        
    with open(qi_file_path, 'w') as fh:
        fh.write(str(min(asf_list)))

def best_eucdist(pset_file_path, qi_file_path, ref_point, weight_vec):
    """   
    Calculate the best distance value in a given point set.

    [Reference] TBA

    Parameters
    ----------
    pset_file_path: file path         
        A file path to load the point set 
    qi_file_path: file path         
        A file path to save the quality indicator value
    ref_point: 1-d float numpy ndarray
        A reference point (an aspiration vector)
    weight_vec: 1-d float numpy ndarray
        A vector to determine a relative importance of each objective function
    """    
    point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)    
    dist_list = []
    for p in point_set:
        dist_list.append(np.linalg.norm(p - ref_point))
        
    with open(qi_file_path, 'w') as fh:
        fh.write(str(min(dist_list)))
        
def expanding_hypercube(pset_file_path_list, qi_file_path_list, ref_point):
    """   
    Calculate the EH metric value of given point sets.

    [Reference] Sunith Bandaru, Henrik Smedberg: A parameterless performance metric for reference-point based multi-objective evolutionary algorithms. GECCO 2019: 499-506

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    """    
    all_point_sets = []
    for pset_file_path in pset_file_path_list:
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)
        
        # Remove duplicate points from each point set to evalute the quality of only unique points
        del_mask = np.full(len(point_set), True)
        for i, p in enumerate(point_set[:-1]):
            for j, q in enumerate(point_set[i+1:], i+1):
                if np.allclose(p, q, atol=1e-12):
                    del_mask[j] = False                    

        point_set = point_set[del_mask]        
        all_point_sets.append(point_set)

    # 1. Remove all dominated solutions from each point set
    all_points = []
    for point_set in all_point_sets:
        all_points.extend(point_set)

    size_point_set = []
    for point_set in all_point_sets:
        size_point_set.append(len(point_set))
    
    # https://esa.github.io/pygmo2/mo_utils.html
    # ndf (list of 1D NumPy int array): the non dominated fronts, dl (list of 1D NumPy int array): the domination list, dc (1D NumPy int array): the domination count, ndr (1D NumPy int array): the non domination ranks
    _, _, _, ndr = pg.fast_non_dominated_sorting(points=all_points)
    
    p_id = 0
    all_nond_point_sets = []
    for point_set in all_point_sets:
        nd_mask = (ndr[p_id:p_id+len(point_set)] == 0)
        p_id += len(point_set)
        all_nond_point_sets.append(np.array(point_set)[nd_mask])        
    
    # 2. Calculate the EH metric value for each point set
    set_areas = []
    set_sizes = []
    for point_set in all_nond_point_sets:
        # The EH value is 0 for the empty point set.
        if len(point_set) == 0:
            set_areas.append(0)
            set_sizes.append(0)
            continue
        
        h_sizes = []
        for p in point_set:
            s = np.max(np.abs(p - ref_point))
            h_sizes.append(s)
        h_sizes = np.sort(np.array(h_sizes))

        area = 0
        prev_hsize = 0
        # The initial i value should be set to 1. Algorithm 3 in the original EH paper describes that i should be initialized by 0. However, when the initial i value = 0, the first point does not contribute to the EH value.
        for i, s in enumerate(h_sizes, 1):
            area += (float(i)/len(point_set)) * (s - prev_hsize)
            prev_hsize = s

        set_areas.append(area)
        set_sizes.append(max(h_sizes))
        
    max_size = max(set_sizes)
    eh_values = []
    for area, size in zip(set_areas, set_sizes):
        if area == 0:
            value = 0
        else:
            value = area + (max_size - size)
        eh_values.append(value)
        
    for qi_file_path, value in zip(qi_file_path_list, eh_values):            
        with open(qi_file_path, 'w') as fh:
            fh.write(str(value))
            
def hv_cf(pset_file_path_list, qi_file_path_list, hv_ref_point, ref_point, roi_radius):    
    """   
    Calculate the HV-CF value of given point sets.

    [Reference] Asad Mohammadi, Mohammad Nabi Omidvar, Xiaodong Li: A new performance metric for user-preference based multi-objective evolutionary algorithms. IEEE Congress on Evolutionary Computation 2013: 2825-2832

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    hv_ref_point: 1-d float numpy ndarray
        A reference point for the hypervolume calculation
    ref_point: 1-d float numpy ndarray
        A reference point
    roi_radius: float
        A radius of the ROI
    """        
    all_point_sets = []
    for pset_file_path in pset_file_path_list:        
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)   
        all_point_sets.append(point_set)
        
    # 1. Construct the composite front 
    all_points = []
    for point_set in all_point_sets:
        all_points.extend(point_set)
    
    # https://esa.github.io/pygmo2/mo_utils.html
    # ndf (list of 1D NumPy int array): the non dominated fronts, dl (list of 1D NumPy int array): the domination list, dc (1D NumPy int array): the domination count, ndr (1D NumPy int array): the non domination ranks
    _, _, _, ndr = pg.fast_non_dominated_sorting(points=all_points)
    
    nd_mask = (ndr == 0)
    composite_front = np.array(all_points)[nd_mask]

    # 2. Determine a preferred region
    distance_list = np.zeros(len(composite_front))
    for i, p in enumerate(composite_front):
        distance_list[i] = np.linalg.norm(p-ref_point)

    mid_id = np.argmin(distance_list)
    mid_point = composite_front[mid_id]

    # 3. Calculate the HV-CF value
    for qi_file_path, point_set in zip(qi_file_path_list, all_point_sets):    
        # Remove points out side the preferred region
        del_mask = np.full(len(point_set), True)
        for i, p in enumerate(point_set):
            d = np.linalg.norm(p-mid_point)
            if d > roi_radius:
                del_mask[i] = False
        trimed_point_set = point_set[del_mask]            

        # Remove points dominated by the HV-reference point 
        del_mask = np.full(len(trimed_point_set), True)    
        for i, p in enumerate(trimed_point_set):        
            # "Return true if obj1 Pareto dominates obj2, false otherwise. Minimization is assumed."
            # https://esa.github.io/pagmo2/docs/cpp/utils/multi_objective.html
            del_mask[i] = pg.pareto_dominance(p, hv_ref_point)
        trimed_point_set = trimed_point_set[del_mask]        
        
        # For some cases, trimed_point_set can be empty. 
        hv_value = 0
        if len(trimed_point_set) > 0:
            hv = pg.hypervolume(trimed_point_set)
            hv_value = hv.compute(hv_ref_point)

        with open(qi_file_path, 'w') as fh:
            fh.write(str(hv_value))    

def igd_cf(pset_file_path_list, qi_file_path_list, ref_point, roi_radius):    
    """   
    Calculate the IGD-CF value of given point sets.

    [Reference] Asad Mohammadi, Mohammad Nabi Omidvar, Xiaodong Li: A new performance metric for user-preference based multi-objective evolutionary algorithms. IEEE Congress on Evolutionary Computation 2013: 2825-2832

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    roi_radius: float
        A radius of the ROI
    """        
    all_point_sets = []
    for pset_file_path in pset_file_path_list:    
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)        
        all_point_sets.append(point_set)
        
    # 1. Construct the composite front 
    all_points = []
    for point_set in all_point_sets:
        all_points.extend(point_set)
    
    # https://esa.github.io/pygmo2/mo_utils.html
    # ndf (list of 1D NumPy int array): the non dominated fronts, dl (list of 1D NumPy int array): the domination list, dc (1D NumPy int array): the domination count, ndr (1D NumPy int array): the non domination ranks
    _, _, _, ndr = pg.fast_non_dominated_sorting(points=all_points)

    nd_mask = (ndr == 0)
    composite_front = np.array(all_points)[nd_mask]

    # 2. Determine a preferred region
    distance_list = np.zeros(len(composite_front))
    for i, p in enumerate(composite_front):
        distance_list[i] = np.linalg.norm(p-ref_point)

    mid_id = np.argmin(distance_list)
    mid_point = composite_front[mid_id]

    # 2.5 All points in the composit front become reference points for the IGD-CF calculation    
    igd_ref_point_set = composite_front
    
    # 3. Calculate the IGD-CF value
    for qi_file_path, point_set in zip(qi_file_path_list, all_point_sets):    
        # Remove points out side the preferred region
        del_mask = np.full(len(point_set), True)
        for i, p in enumerate(point_set):
            d = np.linalg.norm(p-mid_point)
            if d > roi_radius:
                del_mask[i] = False                
        trimed_point_set = point_set[del_mask]

        # For some cases, trimed_point_set can be empty
        igd_value = 1e+20
        if len(trimed_point_set) > 0:
            igd_value = igd(igd_ref_point_set, trimed_point_set)
            
        with open(qi_file_path, 'w') as fh:
            fh.write(str(igd_value))

def r_igd_hv(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, ref_point):
    """   
    Calculate the R-IGD and R-HV values of given point sets. This function calls the rmetric function in pymoo (https://github.com/anyoptimization/pymoo/blob/master/pymoo/indicators/rmetric.
    
    [Reference] Ke Li, Kalyanmoy Deb, Xin Yao: R-Metric: Evaluating the Performance of Preference-Based Evolutionary Multiobjective Optimization Using Reference Points. IEEE Trans. Evol. Comput. 22(6): 821-835 (2018)

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    igd_refpset_file_path: a file path       
        A file path to load the reference point set for the IGD calculation
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    """        
    all_point_sets = []
    for pset_file_path in pset_file_path_list:    
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)        
        all_point_sets.append(point_set)

    # Load a reference point file for the IGD calculation        
    igd_ref_point_set = np.loadtxt(igd_refpset_file_path, delimiter=",", comments="#", dtype=float)  

    # The rmetric function in pymoo requires "problem", but it is used to generate reference points for the IGD calculation only when "pf" is not given. Thus, when "pf" is available, any "problem" is fine.
    n_obj = len(all_point_sets[0][0])
    dummy_problem = get_problem("dtlz2", n_obj=n_obj)    
    
    for i, (qi_file_path, point_set) in enumerate(zip(qi_file_path_list, all_point_sets)):            
        # Aggregate points but the i-th point_set
        other_points = []
        for j, p in enumerate(all_point_sets):
            if i != j:
                other_points.extend(p)                
        other_points = np.array(other_points)

        # "delta" = the length of a hypercube = 2 * radious?
        rmetric = get_performance_indicator("rmetric", problem=dummy_problem, ref_points=np.array([ref_point]), pf=igd_ref_point_set, delta=0.2)
        rigd_value, rhv_value = rmetric.do(point_set, others=other_points, calc_hv=True)

        if rigd_value == None:
            rigd_value = 1e+20
        if rhv_value == None:
            rhv_value = 0
            
        # Save the R-IGD value
        with open(qi_file_path, 'w') as fh:
            fh.write(str(rigd_value))

        # Save the R-HV value      
        rhv_file_path = qi_file_path.replace('r-igd', 'r-hv')
        with open(rhv_file_path, 'w') as fh:
            fh.write(str(rhv_value))

def hvz(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, ref_point):    
    """   
    Calculate the HV_z value of given point sets.

    [Reference] Ana Belen Ruiz, Ruben Saborido, Mariano Luque: A preference-based evolutionary algorithm for multiobjective optimization: the weighting achievement scalarizing function genetic algorithm. J. Glob. Optim. 62(1): 101-129 (2015)

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    igd_refpset_file_path: a file path       
        A file path to load the reference point set for the IGD calculation
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    """    
    # Load a reference point file. The reference point set is used only to determine the position of the reference point for the hypervolume calculation.
    igd_ref_point_set = np.loadtxt(igd_refpset_file_path, delimiter=",", comments="#", dtype=float)
    
    # 1. Set the HV-reference point    
    del_mask = np.full(len(igd_ref_point_set), True)    
    for i, igd_p in enumerate(igd_ref_point_set):        
        # "Return true if obj1 Pareto dominates obj2, false otherwise. Minimization is assumed."
        # https://esa.github.io/pagmo2/docs/cpp/utils/multi_objective.html
        del_mask[i] = pg.pareto_dominance(ref_point, igd_p)

    # When ref_point can dominate any point in igd_ref_point_set, ref_point is infeasible
    if np.any(del_mask):
        dominated_point_set = igd_ref_point_set[del_mask]        
        hv_ref_point = dominated_point_set.max(axis=0)
    else:
        # When ref_point cannot dominate any point in igd_ref_point_set, ref_point is feasible
        hv_ref_point = ref_point    

    # 2. Calculate the hypervolume value with hv_ref_point for each point set 
    for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)        
        del_mask = np.full(len(point_set), True)    
        for i, p in enumerate(point_set):        
            # "Return true if obj1 Pareto dominates obj2, false otherwise. Minimization is assumed."
            # https://esa.github.io/pagmo2/docs/cpp/utils/multi_objective.html
            del_mask[i] = pg.pareto_dominance(p, hv_ref_point)
        point_set = point_set[del_mask]        

        # For some cases, trimed_point_set can be empty. 
        hv_value = 0
        if len(point_set) > 0:        
            hv = pg.hypervolume(point_set)
            hv_value = hv.compute(hv_ref_point)
        
        with open(qi_file_path, 'w') as fh:
            fh.write(str(hv_value))    
            
def med(pset_file_path, qi_file_path, igd_refpset_file_path, ref_point):
    """   
    Calculate the MED value of given point sets.

    [Reference] R. Tang, K. Li, W. Ding, Y. Wang, H. Zhou, and G. Fu, "Reference Point Based Multi-Objective Optimization of Reservoir Operation- a Comparison of Three Algorithms," Water Resour. Manag., vol. 34, pp. 1005-1020, 2020.

    Parameters
    ----------
    pset_file_path: file path         
        A file path to load the point set 
    qi_file_path: file path         
        A file path to save the quality indicator value
    igd_refpset_file_path: file path       
        A file path to load the reference point set for the IGD calculation
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    """        
    # Load the point set found by alg
    point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)        
    # Load the GD/IGD-reference point data. Note that this is used to obtain the maximum and minimum values for each objective.
    igd_ref_point_set = np.loadtxt(igd_refpset_file_path, delimiter=",", comments="#", dtype=float)    
    ideal_point = igd_ref_point_set.min(axis=0)
    nadir_point = igd_ref_point_set.max(axis=0)

    # Calculate the MED value
    sum_dist = 0
    nor_rp = (ref_point - ideal_point) / (nadir_point - ideal_point)    
    for p in point_set:         
        nor_p = (p - ideal_point) / (nadir_point - ideal_point)
        sum_dist += np.linalg.norm(nor_p - nor_rp)
    med_value = sum_dist / len(point_set)

    # Save the result
    with open(qi_file_path, 'w') as fh:
        fh.write(str(med_value))

def igd_p(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, ref_point):
    """   
    Calculate the IGD-P value of given point sets.

    [Reference] W. Luo, L. Shi, X. Lin, and C. A. C. Coello, "The \hat{g}-dominance relation for preference-based evolutionary multi-objective optimization," in IEEE Congress on Evolutionary Computation, CEC 2019, Wellington, New Zealand, June 10-13, 2019. IEEE, 2019, pp. 2418-2425. [Online]. Available: https://doi.org/10.1109/CEC.2019.8790321

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    igd_refpset_file_path: a file path       
        A file path to load the reference point set for the IGD calculation
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    """        
    # Load a reference point file. The reference point set is used only to determine the position of the reference point for the hypervolume calculation.
    igd_ref_point_set = np.loadtxt(igd_refpset_file_path, delimiter=",", comments="#", dtype=float)  
    
    # 1. Set IGD-reference points
    del_mask = np.full(len(igd_ref_point_set), True)    
    for i, igd_p in enumerate(igd_ref_point_set):        
        # "Return true if obj1 Pareto dominates obj2, false otherwise. Minimization is assumed."
        # https://esa.github.io/pagmo2/docs/cpp/utils/multi_objective.html
        del_mask[i] = pg.pareto_dominance(ref_point, igd_p)

    # When ref_point can dominate any point in igd_ref_point_set, ref_point is infeasible
    # When ref_point cannot dominate any point in igd_ref_point_set, ref_point is feasible
    if not np.any(del_mask):
        for i, igd_p in enumerate(igd_ref_point_set):        
            del_mask[i] = pg.pareto_dominance(igd_p, ref_point)
            
    igd_ref_point_set = igd_ref_point_set[del_mask]
    
    if len(igd_ref_point_set) == 0:
        logger.error("The IGD-referfence point set for the IGD-P calculation is empty. Perhaps, the reference point is on the Pareto front. %s", igd_refpset_file_path)
        exit(1)
    
    # 2. Calculate the IGD value
    for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)
        igd_value = igd(igd_ref_point_set, point_set)
        with open(qi_file_path, 'w') as fh:
            fh.write(str(igd_value))

def igd_c(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, ref_point, roi_radius):
    """   
    Calculate the IGD-C value of given point sets.

    [Reference] A. Mohammadi, M. N. Omidvar, X. Li, and K. Deb, "Integrating user preferences and decomposition methods for many-objective optimization," in Proceedings of the IEEE Congress on Evolutionary Computation, CEC 2014, Beijing, China, July 6-11, 2014. IEEE, 2014, pp. 421-428. [Online]. Available: https://doi.org/10.1109/CEC.2014.6900595

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    igd_refpset_file_path: a file path       
        A file path to load the reference point set for the IGD calculation
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    roi_radius: float
        A radius of the ROI
    """    
    # 1. Set IGD-reference points        
    # Load a reference point file. The reference point set is used only to determine the position of the reference point for the hypervolume calculation.
    igd_ref_point_set = np.loadtxt(igd_refpset_file_path, delimiter=",", comments="#", dtype=float)  
    # 1.1 Find the pivot point
    distance_list = np.zeros(len(igd_ref_point_set))
    for i, p in enumerate(igd_ref_point_set):
        distance_list[i] = np.linalg.norm(p - ref_point)

    pivot_id = np.argmin(distance_list)
    pivot_point = igd_ref_point_set[pivot_id]
    
    # 1.2 Find points in the preferred region. The points are used as IGD-reference points.
    del_mask = np.full(len(igd_ref_point_set), False)
    for i, p in enumerate(igd_ref_point_set):
        d = np.linalg.norm(p - pivot_point)
        if d <= roi_radius:
            del_mask[i] = True
    igd_ref_point_set = igd_ref_point_set[del_mask]
    
    if len(igd_ref_point_set) == 0:
        logger.error("The IGD-referfence point set is empty. %s", igd_refpset_file_path)        
        exit(1)

    # 2. Calculate the IGD value
    for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):        
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)        
        igd_value = igd(igd_ref_point_set, point_set)            
        with open(qi_file_path, 'w') as fh:
            fh.write(str(igd_value))
            
def igd_a(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, ref_point, roi_radius, weight_vec):
    """   
    Calculate the IGD-A value of given point sets.

    [Reference] None

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    igd_refpset_file_path: a file path       
        A file path to load the reference point set for the IGD calculation
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    roi_radius: float
        A radius of the ROI
    weight_vec: 1-d float numpy ndarray
        A vector to determine a relative importance of each objective function
    """    
    # 1. Set IGD-reference points        
    # Load a reference point file. The reference point set is used only to determine the position of the reference point for the hypervolume calculation.
    igd_ref_point_set = np.loadtxt(igd_refpset_file_path, delimiter=",", comments="#", dtype=float)  
    # 1.1 Find the pivot point
    distance_list = np.zeros(len(igd_ref_point_set))
    for i, p in enumerate(igd_ref_point_set):
        distance_list[i] = asf1(p, ref_point, weight_vec)            
        
    pivot_id = np.argmin(distance_list)
    pivot_point = igd_ref_point_set[pivot_id]
    
    # 1.2 Find points in the preferred region. The points are used as IGD-reference points.
    del_mask = np.full(len(igd_ref_point_set), False)
    for i, p in enumerate(igd_ref_point_set):
        d = np.linalg.norm(p - pivot_point)
        if d <= roi_radius:
            del_mask[i] = True
    igd_ref_point_set = igd_ref_point_set[del_mask]
    
    if len(igd_ref_point_set) == 0:
        logger.error("The IGD-referfence point set is empty. %s", igd_refpset_file_path)        
        exit(1)

    # 2. Calculate the IGD value
    for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):        
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)        
        igd_value = igd(igd_ref_point_set, point_set)            
        with open(qi_file_path, 'w') as fh:
            fh.write(str(igd_value))

def pr(pset_file_path_list, qi_file_path_list, ref_point):    
    """   
    Calculate the PR value of given point sets.

    [Reference] E. Filatovas, A. Lancinskas, O. Kurasova, and J. Zilinskas, "A preference-based multi-objective evolutionary algorithm R-NSGA-II with stochastic local search," Central Eur. J. Oper. Res., vol. 25, no. 4, pp. 859-878, 2017. [Online]. Available: https://doi.org/10.1007/s10100-016-0443-x

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    """    
    for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)                
        # 1. Trim the point set
        del_mask = np.full(len(point_set), True)    
        for i, p in enumerate(point_set):        
            # "Return true if obj1 Pareto dominates obj2, false otherwise. Minimization is assumed."
            # https://esa.github.io/pagmo2/docs/cpp/utils/multi_objective.html
            del_mask[i] = pg.pareto_dominance(ref_point, p)
            # When ref_point can dominate any point in igd_ref_point_set, ref_point is infeasible
            # When ref_point cannot dominate any point in igd_ref_point_set, ref_point is feasible
            if not np.any(del_mask):
                for i, p in enumerate(point_set):        
                    del_mask[i] = pg.pareto_dominance(p, ref_point)

        pr_value = 100 * (float(np.count_nonzero(del_mask)) / len(point_set))
        with open(qi_file_path, 'w') as fh:
            fh.write(str(pr_value))

def pmod(pset_file_path_list, qi_file_path_list, ref_point, roi_radius, penalty_value=1.5):    
    """   
    Calculate the PMOD value of given point sets.

    [Reference] Zhanglu Hou, Shengxiang Yang, Juan Zou, Jinhua Zheng, Guo Yu, Gan Ruan: A Performance Indicator for Reference-Point-Based Multiobjective Evolutionary Optimization. SSCI 2018: 1571-1578

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    roi_radius: float
        A radius of the ROI
    penalty_value: float
        A penalty value for points outside the ROI
    """    
    unit_ref_vec = ref_point / np.linalg.norm(ref_point)

    for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)                

        # 1. Map the points to a hyperplane
        mapped_pset = []
        for p in point_set:            
            dist = np.dot(ref_point - p, unit_ref_vec)
            q = p + (dist * unit_ref_vec)
            mapped_pset.append(q)

        # 2. Calculate the D1 value, which is the Euclidean between each mapped point and the reference point
        d1_arr = np.zeros(len(mapped_pset))
        for i, p in enumerate(mapped_pset):
            d1_arr[i] = np.linalg.norm(p - ref_point)
        
        # Obtain the minimum L1 norm between each mapped point and another one
        min_l1_arr = np.zeros(len(mapped_pset))
        for i, p in enumerate(mapped_pset):
            min_l1_arr[i] = 1e+20
            for j, q in enumerate(mapped_pset):
                if i != j:
                    l1_value = np.linalg.norm(p - q, ord=1)
                    if l1_value < min_l1_arr[i]:
                        min_l1_arr[i] = l1_value

        # 3. Calculate the D2 value, which is the unbiased standard deviation of "min_l1_arr"
        d2_value = np.std(min_l1_arr, ddof=1)

        # 4.  Calculate the D3 value, which is the Euclidean distance between each (unmapped) point and the origin, i.e., the L2 norm of the (unmapped) point
        d3_arr = np.zeros(len(point_set))
        for i, p in enumerate(point_set):
            d3_arr[i] = np.linalg.norm(p)
            # If  the point is outside of the ROI, it is penalized
            if d1_arr[i] <= roi_radius:
                d3_arr[i] *= penalty_value                

        # 5. Calculate the PMOD value based on D1, D2, and D3
        pmod_value = ((1./ len(mapped_pset)) * (np.sum(d1_arr[i]) + np.sum(d3_arr[i]))) + d2_value 
        with open(qi_file_path, 'w') as fh:
            fh.write(str(pmod_value))

def angle(p, q):
    pq = np.inner(p, q)
    norm_p = np.linalg.norm(p)
    norm_q = np.linalg.norm(q)    
    return np.arccos(pq / (norm_p * norm_q))
    #return np.arccos(pq / (norm_p * norm_q)) * 180 / np.pi
            
def pmda(pset_file_path_list, qi_file_path_list, ref_point, epsilon=0.1, penalty_value=1./np.pi):    
    """   
    Calculate the PMDA value of given point sets.

    [Reference] Guo Yu, Jinhua Zheng, Xiaodong Li: An improved performance metric for multiobjective evolutionary algorithms with user preferences. CEC 2015: 908-915

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    ref_point: 1-d float numpy ndarray
        An "n_obj"-dimensional reference point
    epsilon: float
        A parameter to control the positions of light beams
    penalty_value: float
        A penalty value for points outside the ROI
    """    
    n_obj = len(ref_point)
    if n_obj > 2:
        logger.error("The present implementation of PMDA can handle only two-objective optimization.")
        exit(1)        
    
    int_point_set = np.identity(n_obj)
    base_axis = int_point_set[0]
    neg_base_axis = -int_point_set[0]
    
    r1 = (1. / np.sum(ref_point)) * ref_point
    
    Q = []
    for ip in int_point_set:
        p = r1 + (epsilon * (ip - r1))
        Q.append(p)
    Q.append(r1)
       
    angle_set = []
    for q in Q[:-1]:
        theta = angle(q, base_axis)
        angle_set.append(theta)
    
    all_point_sets = []
    for pset_file_path in pset_file_path_list:    
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)        
        all_point_sets.append(point_set)

    all_point_sets_in_roi = []
    for pset in all_point_sets:            
        for p in pset:                    
            # Determine whether p is in the prefered region
            # TODO: This is only for bi-objective optimization
            theta = angle(p, base_axis)
            if angle_set[0] <= theta and theta <= angle_set[1]:            
                all_point_sets_in_roi.append(p)
                    
    delta = np.array(all_point_sets_in_roi).min()
        
    Q2 = []
    for q in Q:
        q2 = delta * q
        Q2.append(q2)

    for pset, qi_file_path in zip(all_point_sets, qi_file_path_list):                    
        sum_d_theta = 0
        for i, p in enumerate(pset):
            d = np.min(np.linalg.norm(p-Q2, axis=1))
            # Determine whether p is in the prefered region
            # TODO: This is only for bi-objective optimization
            theta = angle(p, base_axis)
            if angle_set[0] <= theta and theta <= angle_set[1]:            
                theta = 0
            else:
                theta = penalty_value * angle(p, Q2[-1])
            
            sum_d_theta += d + theta
            
        pmda_value = sum_d_theta / len(pset)
        with open(qi_file_path, 'w') as fh:
            fh.write(str(pmda_value))
            
def pure_igd(pset_file_path_list, qi_file_path_list, igd_refpset_file_path):
    """   
    Calculate the "pure" IGD value of given point sets.

    [Reference] C. A. C. Coello and M. R. Sierra, "A Study of the Parallelization of a Coevolutionary Multi-objective Evolutionary Algorithm," in MICAI, 2004, pp. 688-697.

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    igd_refpset_file_path: a file path       
        A file path to load the reference point set for the IGD calculation
    """        
    # Load a reference point file.
    igd_ref_point_set = np.loadtxt(igd_refpset_file_path, delimiter=",", comments="#", dtype=float)  
        
    # 2. Calculate the IGD value       
    for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):    
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)
        igd_value = igd(igd_ref_point_set, point_set)            
        with open(qi_file_path, 'w') as fh:
            fh.write(str(igd_value))

def pure_igdplus(pset_file_path_list, qi_file_path_list, igd_refpset_file_path):
    """   
    Calculate the "pure" IGD^+ value of given point sets.

    [Reference] Hisao Ishibuchi, Hiroyuki Masuda, Yuki Tanigaki, Yusuke Nojima: Modified Distance Calculation in Generational Distance and Inverted Generational Distance. EMO (2) 2015: 110-125

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    igd_refpset_file_path: a file path       
        A file path to load the reference point set for the IGD calculation
    """        
    # Load a reference point file.
    igd_ref_point_set = np.loadtxt(igd_refpset_file_path, delimiter=",", comments="#", dtype=float)  
        
    # 2. Calculate the IGD value       
    for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):    
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)
        igdp_value = igdplus(igd_ref_point_set, point_set)            
        with open(qi_file_path, 'w') as fh:
            fh.write(str(igdp_value))    
            
def pure_hv(pset_file_path_list, qi_file_path_list, hv_ref_point):
    """   
    Calculate the "pure" HV value of given point sets.

    [Reference] E. Zitzler and L. Thiele, "Multiobjective optimization using evolutionary algorithms - A comparative case study," in Parallel Problem Solving from Nature - PPSN V, 5th International Conference, 1998

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    hv_ref_point: 1-d float numpy ndarray
        A reference point for the hypervolume calculation
    """                
    for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):    
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)
        
        del_mask = np.full(len(point_set), True)    
        for i, p in enumerate(point_set):
            # "Return true if obj1 Pareto dominates obj2, false otherwise. Minimization is assumed."
            # https://esa.github.io/pagmo2/docs/cpp/utils/multi_objective.html        
            del_mask[i] = pg.pareto_dominance(p, hv_ref_point)
        point_set = point_set[del_mask]
        
        hv = pg.hypervolume(point_set)
        hv_value = hv.compute(hv_ref_point)    
        with open(qi_file_path, 'w') as fh:
            fh.write(str(hv_value))

def pure_r2(pset_file_path_list, qi_file_path_list, weight_vec_file_path, ideal_point):
    """   
    Calculate the "pure" R2 value of given point sets.

    [Reference] M. P. Hansen and A. Jaszkiewicz, "Evaluating the quality of approximations to the non-dominated set," Poznan University of Technology, Tech. Rep. IMM-REP-1998-7, 1998.

    Parameters
    ----------
    pset_file_path_list: a list of file paths         
        A list of file paths to load the point sets 
    qi_file_path: a list of  file paths         
        A list of file paths to save the quality indicator values
    weight_vec_file_path: a file path       
        A file path to load the weight vector set for the R2 calculation
    """        
    # Load a reference point file.
    weight_vec_set = np.loadtxt(weight_vec_file_path, delimiter=",", comments="#", dtype=float)  
        
    # 2. Calculate the IGD value       
    for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):    
        point_set = np.loadtxt(pset_file_path, delimiter=",", comments="#", dtype=float)
        r2_value = r2(weight_vec_set, ideal_point, point_set)            
        with open(qi_file_path, 'w') as fh:
            fh.write(str(r2_value))
            
def calc_qindicator(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, n_obj, ref_point, weight_vec, qindicator, hv_ref_point, roi_radius=0.1):    
    if qindicator == 'masf':
        for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):
            minimum_asf(pset_file_path, qi_file_path, ref_point, weight_vec)            
    elif qindicator == 'eh':
        expanding_hypercube(pset_file_path_list, qi_file_path_list, ref_point)        
    elif qindicator == 'hv-cf':
        hv_cf(pset_file_path_list, qi_file_path_list, hv_ref_point, ref_point, roi_radius)
    elif qindicator == 'igd-cf':
        igd_cf(pset_file_path_list, qi_file_path_list, ref_point, roi_radius)
    elif qindicator == 'r-igd':
        r_igd_hv(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, ref_point)       
    elif qindicator == 'hvz':
        hvz(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, ref_point)     
    elif qindicator == 'med':
        for pset_file_path, qi_file_path in zip(pset_file_path_list, qi_file_path_list):
            med(pset_file_path, qi_file_path, igd_refpset_file_path, ref_point)            
    elif qindicator == 'igd-p':
        igd_p(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, ref_point)
    elif qindicator == 'igd-c':
        igd_c(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, ref_point, roi_radius)
    elif qindicator == 'igd-a':
        igd_a(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, ref_point, roi_radius, weight_vec)
    elif qindicator == 'pr':
        pr(pset_file_path_list, qi_file_path_list, ref_point)
    elif qindicator == 'igd':
        pure_igd(pset_file_path_list, qi_file_path_list, igd_refpset_file_path)
    elif qindicator == 'igdplus':
        pure_igdplus(pset_file_path_list, qi_file_path_list, igd_refpset_file_path)
    elif qindicator == 'hv':
        pure_hv(pset_file_path_list, qi_file_path_list, hv_ref_point)
    elif qindicator == 'r2':
        ideal_point = np.full(n_obj, 0)
        pure_r2(pset_file_path_list, qi_file_path_list, weight_vec_file_path, ideal_point)        
    elif qindicator == 'pmod':
        pmod(pset_file_path_list, qi_file_path_list, ref_point, roi_radius)        
    elif qindicator == 'pmda':
        pmda(pset_file_path_list, qi_file_path_list, ref_point)        
    else:
        logger.error("%s is not defined.", qindicator)
        exit(1)
                                            
if __name__ == '__main__':
    n_obj = 2
    weight_vec = np.full(n_obj, 1.0/n_obj)
    ref_point = np.array([0.5, 0.5])
    hv_ref_point = np.full(n_obj, 1.1)
    
    # Select one from ['masf', 'med', 'igd-c', 'igd-a', 'igd-p', 'hvz', 'pr', 'pmod', 'igd-cf', 'hv-cf', 'pmda', 'r-igd', 'eh', 'hv', 'igd']. Please note that R-HV and R-IGD are calculated simultaneously.    
    qindicator = 'masf'
    
    # File path for the IGD-reference point set. The following quality indicators need it: 'r-igd', 'hvz', 'med', 'igd-p', 'igd-c', 'igd-a', 'igd', and 'igdplus'. If you consider another problem, please prepare the IGD-reference point set.
    problem = 'DTLZ2' 
    igd_refpset_file_path = './ref_point_dataset/{}_d{}_n1000.csv'.format(problem, n_obj)
    # Only for R2
    weight_vec_file_path = os.path.join('./weight_point_dataset', 'd{}_n{}.csv'.format(n_obj, 1000))

    # File paths for point sets to be compared.
    pset_file_path_list = []
    for pid in range(10):
        tmp = './pset_dataset/{}_d{}_np20_pid{}.csv'.format(problem, n_obj, pid)
        pset_file_path_list.append(tmp)           

    # Directory paths to save the QI values. 
    pp_res_dir_path = os.path.join('./pp_results', qindicator)
    os.makedirs(pp_res_dir_path, exist_ok=True)
    # R-HV and R-IGD are calculated simultaneously.    
    if qindicator == 'r-igd':
        pp_res_dir_path = os.path.join('./pp_results', 'r-hv')
        os.makedirs(pp_res_dir_path, exist_ok=True)

    # File paths to save the QI values. Any file name is fine, including '1.csv', '2.csv', ...
    qi_file_path_list = []
    str_ref_point =  ref_point.astype(str)
    str_ref_point = '_'.join(str_ref_point)
    for pid in range(10):
        file_path = os.path.join(pp_res_dir_path, 'z{}_{}_d{}_np20_pid{}.csv'.format(str_ref_point, problem, n_obj, pid))
        qi_file_path_list.append(file_path)
    
    calc_qindicator(pset_file_path_list, qi_file_path_list, igd_refpset_file_path, n_obj, ref_point, weight_vec, qindicator, hv_ref_point, roi_radius=0.1)
