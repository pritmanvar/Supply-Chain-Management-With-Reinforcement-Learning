from inference_storage import get_storage_insights, get_storage_graphs
from inference_manufucturing_2 import get_manufacturing_graphs
import json
# Storage inputs

with open('data.json') as json_file:
    data = json.load(json_file)

print()
print("Data")
print(data)
    
# date = input("Enter the date in this yyyy-mm-dd format:  ")
# manufacturing_price = int(input("Enter the Manufacturing Price : "))
# selling_price = int(input("Enter the Selling price: "))

# manufucturing_data = {}
# # Manufacturing inputs
# manufucturing_data['raw_material_cost'] = int(input("Enter the raw material cost: "))
# manufucturing_data['main_cost'] = int(input("Enter the sum of storage cost and manufacturing cost per day: "))
# manufucturing_data['production_cost_per_product'] = int(input("Enter the production cost per product: "))
# manufucturing_data['max_manufacturing_rate'] = int(input("Enter the max manufucturing rate per hour: "))
# manufucturing_data['product_capacity'] = int(input("Enter the manufuctured product capacity: "))
# manufucturing_data['selling_price'] = int(input("Enter the selling price of the product: "))
# manufucturing_data['raw_material_capacity'] = int(input("Enter the raw material capacity: "))
# manufucturing_data['require_raw_material_per_product'] = int(input("Enter the required raw material per product: "))

# manufucturing_data['vehicals_capacity'] = str(input("Enter capacity of your unique vehicles seperated by space. -> capacity_for_v1 capacity_for_v2 ")).split(" ")
# manufucturing_data['vehicals_cost'] = str(input("Enter cost of each vehical per KM seperated by space. -> cost_for_v1 cost_for_v2 ")).split(" ")
# manufucturing_data['num_of_vehicals_per_type'] = str(input("Enter number of vehicals instance for each type seperated by space. -> count_for_v1 count_for_v2 ")).split(" ")
# manufucturing_data['soucre_address'] = str(input("Enter source address: "))
# manufucturing_data['destination_address'] = str(input("Enter destination address: "))


date = data['date']
manufacturing_price = data['manufacturing_price']
selling_price = data['selling_price']
storage_capacity = data['storage_capacity_of_warehouse']
storage_cost = data['storage_cost_of_warehouse']

manufucturing_data = {}
# Manufacturing inputs
manufucturing_data['raw_material_cost'] = data['raw_material_cost']
manufucturing_data['main_cost'] = data['main_cost']
manufucturing_data['production_cost_per_product'] = data['production_cost_per_product']
manufucturing_data['max_manufacturing_rate'] = data['max_manufacturing_rate']
manufucturing_data['product_capacity'] = data['product_capacity']
manufucturing_data['selling_price'] = data['man_selling_price']
manufucturing_data['raw_material_capacity'] = data['raw_material_capacity']
manufucturing_data['require_raw_material_per_product'] = data['require_raw_material_per_product']

manufucturing_data['vehicals_capacity'] = data['vehicals_capacity']
manufucturing_data['vehicals_cost'] = data['vehicals_cost']
manufucturing_data['num_of_vehicals_per_type'] = data['num_of_vehicals_per_type']
manufucturing_data['source_address'] = data['source_address']
manufucturing_data['destination_address'] = data['destination_address']

print()
print("Man data")
print(manufucturing_data)

get_storage_insights(date, manufacturing_price, selling_price, storage_capacity, storage_cost, manufucturing_data)

storage_graph = get_storage_graphs()
manufacturing_graph = get_manufacturing_graphs()

storage_graph.show()
manufacturing_graph.show()