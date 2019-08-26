"""
To solve the linear optimization problem the following software is used:
    - Python > 3.0: www.python.org
    - Pyomo: http://www.pyomo.org/
    - The linear solver CBC: https://projects.coin-or.org/Cbc

Note: The installation folders of Python and the CBC solver have to be added to PATH (environmental variables).

The optimization takes place inside the script but can be also done via the command window using
pyomo solve --solver-manager=neos --solver=cbc diet1.py diet.dat.
In the command window example the solver is not installed locally but the solver capacity of neos-server.org
is used to solve the problem.

The biggest advantage of using in script optimization is that the results of the optimization is loaded back into the
model for further processing like export or aggregation of information. Furthermore the parameters can be changed
iteratively which can be helpful in a sensitive analysis.
"""

from pyomo.environ import *
from pyomo.opt import SolverFactory

# the value for infinity is defined
infinity = float('inf')

"""
Create the model:
    - Here an abstract model is used. The abstract model allows for separation of the input parameters values and
    the model declaration.

        For further information about the difference about an abstract and concrete model
        start here: https://software.sandia.gov/downloads/pub/pyomo/PyomoOnlineDocs.html#_abstract_versus_concrete_models

    -
"""

# model declaration
model = AbstractModel()

"""
Create the main sets that are used to organize the data structure of the model.
All parameters and constraints are created in respect to these sets.
The main parts of the model are
    - YEARS: 2009 - 2014
    - BLOCKS: The demand blocks 1 to 5
    - SUPPLIER: McNeil, Contracts, VCE_GT, Market, VCE_Wind
    - DSM: These are the nine available dsm options
"""
model.BLOCKS = Set()
model.SUPPLIER = Set()
model.YEARS = Set()
model.DSM = Set()

"""
    PARAMETERS
    Parameters are static.
    They can be initialized with different parameters:
        - within: NonNegativeReals -> Meaning that the parameters value must be non negative
                  Boolean -> The parameters values is either 0 or 1
        - default: 0.0 -> A default value that is sometimes necessary if the parameter is referenced in a constrained
         before the model data is loaded into the abstract model
        - mutable: True -> If mutable is set to true the parameter can be changed after the initial data is loaded into
        the model. This is helpful if one wants to run tests over a range of input values.

    For further information see:
        https://software.sandia.gov/downloads/pub/pyomo/PyomoOnlineDocs.html#_parameters
"""

"""
The basic initial parameters for supply cost, demand cost and load hours taken from the case.
"""
# Note: The Parameters is initialized with reference the the supplier and block sets.
# That means we create a supply cost per supplier and block because
# for each supplier there is a different supply cost in each block.
model.supply_cost = Param(model.SUPPLIER, model.BLOCKS)
model.demand = Param(model.BLOCKS)
model.load_hours = Param(model.BLOCKS)

"""
Parameters concerning the supplier:
    - cost increase: The increase in cost for the production of one MWh of electricity. e.g. 1.04 for 4% increase
    - carbon emissions: The carbon emission associated with the production of one MWh of electricity
    - available supply: The supply capacity of each supplier. (Is the same for each block because in this case average
    capacities are used based on the yearly load factor of the power plant - import for Wind or PV.)
    - minimal supply: This parameters is always 0. Just used for the constraint for the purchase of power from a plant.
    (This parameter is not specified in the initial data because it is always 0)
    - implemented: This parameters specifies if the power plant is available or not. Important for the wind power plant.
    (Only the data for the wind power plant is specified in the initial data because for all other suppliers it is just 1).
"""
# parameters concerning the supplier:
# cost increase per year, carbon emission, available supply, if the supply is implermented
model.cost_increase = Param(model.SUPPLIER, within=NonNegativeReals, default=1.0)
model.carbon_emissions = Param(model.SUPPLIER, within=NonNegativeReals)
model.available_supply = Param(model.SUPPLIER, within=NonNegativeReals)
model.minimal_supply = Param(model.SUPPLIER, within=NonNegativeReals, default=0.0)
# the implementation is handled in the bounds for the buy_bounds
model.implemented = Param(model.SUPPLIER, model.YEARS, within=Boolean, default=True)


"""
Parameters concerning cost:
    - discount factor: discounting costs in the future
"""
model.discount_factor = Param(model.YEARS)

"""
Paramter for dms options
    - dsm_max: instead of using a boolean variable for the dsm options a range was introduced to give some flexibility to
     the model. If the dsm_max is 1 it acts in the same way as a boolean constraint. If it is higher than 1 more money
     can be spend on one dsm option.

     Note: This implementation is coarse grained. Only multiples of the initial money can be spend on each dsm option.
            One could make a better model in which the money spend for the dsm options is bounded and the savings for
            each dsm options are related to the money spend. So instead of having 0.34MWh saved on has to use
            0.34 MWh / initial cost in $.

     Note: For bounds (0.0, 10.0) all money is used in one DSM ('Retail Products': 9) @ a budget of 800,000
        -> retail products is the most efficient measure

     Note: How could it be modelled that doubling the money for one DSM doesn't double the savings? It is possible but
     this leads to a non linear model.

     - dsm_min: 0
     - dsm_implemented_param: initial starting conditions. no dsm is implemented. (The data is specified in the initial
     data but doesn't have to be. instead a default value could be used.)

     - dsm_direct cost: the cost for implementing one dsm option
     - dsm_MW_savings: how much MWh hours are saved for each dsm in each block
     - dsm_budget_cap: The budget that can be spend on the dsm options. Initially this is 800.000.
     Note: This option is set mutable to be able to test if a change in the dsm budget constraint changes the overall
     performance.
"""
model.dsm_max = Param(mutable=True)
model.dsm_min = Param(initialize=0)
model.dsm_implemented_param = Param(model.DSM, within=Boolean)
model.dsm_direct_cost = Param(model.DSM, within=NonNegativeReals)
model.dsm_MW_savings = Param(model.DSM, model.BLOCKS, within=NonNegativeReals)
# Budget
model.dsm_budget_cap = Param(mutable=True)

"""
Parameters for emission cap
    - emission_cap: This parameter can be used to constrain the overall allowed emissions. Making it possible to
    optimize for the cost with respect to a certain emission constraint.
    Note: If the emission cap is set high enough (in our case > 1,000,000) the cap doesn't impose a constraint anymore
    because the maximum emissions are lower.
"""
model.emission_cap = Param(mutable=True)
model.emission_minimum = Param(initialize=0.0)

"""
Functions: Variable bounds
These functions return the upper and lower bound the associated variables.
"""

def dsm_bounds(model):
    return model.dsm_min, model.dsm_max

def emission_bounds(model):
    return model.emission_minimum, model.emission_cap

# define the boundary conditions for the supply.
# The power bought cannot exceed the available power supply of the supplier.
def buy_bounds(model, year, supplier, blocks):
    if model.implemented[supplier, year]:
        return model.minimal_supply[supplier], model.available_supply[supplier]
    else:
        return model.minimal_supply[supplier], model.minimal_supply[supplier]

"""
    VARIABLES
"""

"""
    Variables are the components that are computed during the optimization process.
    The following variables have been specified:
        - dsm_implemented: the implemented dsm options (not a Boolean variable here)

        - buy: how much energy is bought from what supplier in which block for every year
        - actual_demand: the initial demand per block is lowered by the implementation of the dsm.
        - total_costs: single value. used for optimization in respect to the total energy cost

        - emissions: the total emissions for each supplier and year
        - total_emissions: single value. used for optimization in respect to the total emissions associated with the
        production of the bought energy
"""

# dsm
model.dsm_implemented = Var(model.DSM, domain=NonNegativeIntegers, bounds=dsm_bounds, initialize=0)

# demand, cost
model.buy = Var(model.YEARS, model.SUPPLIER, model.BLOCKS, bounds=buy_bounds)
model.actual_demand = Var(model.YEARS, model.BLOCKS, domain=NonNegativeReals, initialize=0.0)
model.total_costs = Var(domain=NonNegativeReals)

# emissions
model.emissions = Var(model.SUPPLIER, model.YEARS, domain=NonNegativeReals, bounds=(0.0, infinity))
model.total_emissions = Var(domain=NonNegativeReals, bounds=emission_bounds)

"""
Constraints
"""

# Constraint: emissions per year and supplier
def emission_rule(model, supplier, year):
    """
    The emissions variable is just there to save the total emissions per supplier and year.
    The optimization result doesn't depend on this constraint
    :param model: model reference
    :param supplier: reference to supplier set
    :param year: reference to year set
    :return:
    """
    total_emission = 0

    total_emission = sum(model.buy[year, supplier, block] * model.carbon_emissions[supplier] * model.load_hours[block] for block in model.BLOCKS)

    return model.emissions[supplier, year] == total_emission

model.emissions_constraint = Constraint(model.SUPPLIER, model.YEARS, rule=emission_rule)


# Constraint: total costs
def total_costs_rule(model):
    """
    The total_costs variable saves the total costs.
    The variable and it's constraint just exist due to convenience for later data extraction if one chooses not to
    optimize in respect to total costs.
    :param model: model reference
    :return:
    """
    costs = cost_objective(model)

    return model.total_costs == costs

model.total_costs_constraint = Constraint(rule=total_costs_rule)



def construct_constraint(year, index):
    """
    This function is a convenience function that returns a valid demand constraint for one year.
    The wrapper function construct_constraint is needed because we need a reference to year in the wrapped
    meet_demand function in model.buy[year, supplier, block] and model.implemented[supplier, year]
    :param year: reference to year
    :param index: index of the year. The first year is 2009 so the index is 0. For 2010 the index is 1.
    :return:
    """

    def meet_demand(model, block):
        """
        Enforce the meeting of the demand
        :param model:
        :param block:
        :return:
        """

        # dsm_savints = (YEAR - 2008) * total initial savings for the implemented dsms
        # the first part (index+1) == (YEAR - 2008) because in our model the dsm options reduce the demand not only once
        # but in every consecutive year. So we have a cumulative saving over the years.
        dsm_savings = (index+1)*sum(model.dsm_implemented[dsm] * model.dsm_MW_savings[dsm, block] for dsm in model.DSM)

        # demand - savings by dsm == supply <=> demand == supply + savings_by_dsm
        return (model.demand[block]) == sum(model.buy[year, supplier, block] for supplier in model.SUPPLIER if model.implemented[supplier, year]) + dsm_savings

    return meet_demand

# construct a constraint for each year.
for index, year in enumerate(range(2009, 2015)):
    constraint_fct = construct_constraint(year, index)

    # constraints can be dynamically added by setattr(...)
    # setattr(...) is a general way to add attributes to python objects
    setattr(model, 'limit' + str(year), Constraint(model.BLOCKS, rule=constraint_fct))


# Constraint: DSM budget cap
def dsm_budget_constraint_rule(model):
    """
    The direct cost of the imnplemented DSM shall not exceed the DSM budget cap
    :param model:
    :return:
    """
    return model.dsm_budget_cap >= sum(model.dsm_implemented[dsm] * model.dsm_direct_cost[dsm] for dsm in model.DSM)


# init demand side response (DSM) budget constraint
model.dsm_budget_constraint = Constraint(rule=dsm_budget_constraint_rule)


# Constraint: Total Emissions
def total_emissions_rule(model):
    """
    Constraint to keep track of total emissions
    :param model:
    :return:
    """
    return model.total_emissions == sum(model.emissions[supplier, year] for year in model.YEARS for supplier in model.SUPPLIER)


# init total emission constraint
model.total_emissions_constraint = Constraint(rule=total_emissions_rule)


# Constraint: actual demand
def actual_demand_rule(model, year, block):
    """
    Constraint to keep track of actual demand. The demand is lowered because of the implemented DSM options.
    The savings of the DSM options are cumulative in our case. So each year the demand is lowered by the amount of the
    DSM option savings specified in the table ...
    :param model: model reference
    :return:
    """
    starting_year = 2009
    index = year - starting_year

    dsm_savings = (index+1)*sum(model.dsm_implemented[dsm] * model.dsm_MW_savings[dsm, block] for dsm in model.DSM)

    return model.actual_demand[year, block] == model.demand[block] - dsm_savings


# init actual demand constraint
model.actual_demand_constraint = Constraint(model.YEARS, model.BLOCKS, rule=actual_demand_rule)


"""
Objectives:
"""


# objective: Minimize the total cost
def cost_objective(model):
    """
    The cost objective calculates the total cost.
    The total_cost_per_year = discount_factor * price_increment * how_much_bought_from_whom * load_hours
    :param model: model reference
    :return: Total cost
    """
    starting_year = 2009
    costs = []
    for year in model.YEARS:
        index = year-starting_year
        for supplier in model.SUPPLIER:
            for block in model.BLOCKS:
                cost = model.discount_factor[year] * \
                       pow(model.cost_increase[supplier], index) * \
                       model.buy[year, supplier, block] * \
                       model.supply_cost[supplier, block] * \
                       model.load_hours[block]

                costs.append(cost)

    return sum(costs)


def emission_objective(model):
    """
    The emission objective calculates the total emissions.
    the total emissions are equal to the amount of energy bought from which supplier * load hours * carbon emissions per MWh
    :param model: model reference
    :return: total emissions
    """
    total_emissions = []
    for year in model.YEARS:
        for supplier in model.SUPPLIER:
            for block in model.BLOCKS:
                emission = model.buy[year, supplier, block] * model.carbon_emissions[supplier] * model.load_hours[block]

                total_emissions.append(emission)

    return sum(total_emissions)

"""
Only one objective can be in place at a time.
"""
# Optimize in regards to costs
model.total_cost_obj = Objective(rule=cost_objective, sense=minimize)

# Optimize in regards to emissions
# model.total_emissions_obj = Objective(rule=emission_objective, sense=minimize)


"""
Auxiliary Functions
"""
def construct_dict(model):
    """
    Convenience function to construct a dictionary from the model data to aggregate the data.
    :param model:
    :return:
    """
    results = {
        'total_emissions': value(model.total_emissions),
        'energy_costs': value(model.total_costs),
        'merit_order': {},
        'emissions': {},
        'implemented_dsm': {},
        'demand': {},
        'dsm_amount': 0,
        'dsm_costs': 0,
        'total_costs': 0,
        'MW_saved_by_DSM': 0,
        'emissions': {},
        'total_energy_consumed_MWh': 0,
        'emission_per_MWh': 0,
    }


    # create merit-order
    for year in model.YEARS:
        results['merit_order'][year] = {}

        for supplier in model.SUPPLIER:
            results['merit_order'][year][supplier] = [{'value': value(model.buy[year, supplier, block]), 'name': block}
                                                         for block in model.BLOCKS]

    # create emissions overview
    for year in model.YEARS:
        results['emissions'][year] = {}

        for supplier in model.SUPPLIER:
            results['emissions'][year][supplier] = value(model.emissions[supplier, year])

    # implemented dsm
    # calculate cost for all implemented dsm
    for dsm in model.DSM:
        results['implemented_dsm'][dsm] = value(model.dsm_implemented[dsm])
        results['dsm_costs'] += value(model.dsm_implemented[dsm])*value(model.dsm_direct_cost[dsm])

        # amount of dsm implemented
        if results['implemented_dsm'][dsm]:
            results['dsm_amount'] += 1

    # total cost = energy costs + costs for dsm
    results['total_costs'] = results['energy_costs'] + results['dsm_costs']

    # power saved in MW saved by the DSM
    for dsm in model.DSM:
        for block in model.BLOCKS:
            results['MW_saved_by_DSM'] += value(model.dsm_implemented[dsm])*value(model.dsm_MW_savings[dsm, block])

    # demand
    for year in model.YEARS:
        results['demand'][year] = {}
        for block in model.BLOCKS:
            results['demand'][year][block] = value(model.actual_demand[year, block])

    # total energy consumed
    total_energy_consumed = 0
    for year in model.YEARS:
        for block in model.BLOCKS:
            total_energy_consumed += results['demand'][year][block] * value(model.load_hours[block])

    results['total_energy_consumed_MWh'] = total_energy_consumed

    # emission per MWh
    results['emission_per_MWh'] = results['total_emissions'] / results['total_energy_consumed_MWh']

    return results


"""
Solve
"""
# Create a solver instance by
optsolver = SolverFactory('cbc')

# specify budgets to use for the DSM budget constraint
budgets = [500000, 600000, 700000, 800000, 850000, 900000, 925000, 950000, 975000, 1000000]

budget_result = {}   # the result for one budget
result_list = []    # the results for all budgets
print('budgets', budgets)
for budget in budgets:
    # Create instance with the data specified in a file
    instance = model.create_instance('vermont_city_with_dsm_emission_cap.dat')

    # Set the budget cap
    instance.dsm_budget_cap = budget

    # send them to the solver(s)
    results = optsolver.solve(instance, tee=True)

    # display the results
    instance.display()

    # aggregate data
    budget_result[budget] = construct_dict(instance)
    budget_result[budget]['dsm_budget'] = float(str(budget))
    result_list.append(budget_result[budget])

# print information to the console
for budget in budgets:
    print(budget)
    print('energy cost:', budget_result[budget]['energy_costs'])
    print('total emissions:', budget_result[budget]['total_emissions'])
    print('dsm budget', budget_result[budget]['dsm_budget'])
    print('dsm cost:', budget_result[budget]['dsm_costs'])
    print('dsm implemented(amount):', budget_result[budget]['dsm_amount'])
    print('MW saved by DSMs', budget_result[budget]['MW_saved_by_DSM'])
    print('dsm implemented:', budget_result[budget]['implemented_dsm'])
    print('total costs', budget_result[budget]['total_costs'])
    print('total energy consumed', budget_result[budget]['total_energy_consumed_MWh'])
    print('emission tCO2 per MWh', budget_result[budget]['emission_per_MWh'])
    print()

"""
To look which budget is the cheapest the total cost is used.
The total cost = energy cost + costs for dsm
"""
result_list.sort(key=lambda result: result['total_costs'])

for result in result_list:
    print(result['dsm_budget'])


