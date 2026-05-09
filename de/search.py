from de.awgn import findThresForRecipe as GA_findThresForRecipe
from de.bec import findThresForRecipe as BEC_findThresForRecipe
from de.recipe import isRecipeSatisfyRect, find_composition
# recipe: A map from column type to the number of columns of that type
#   where column type is the weights of each block in that column.

# book: A map from degree to the number of nodes with that degree

if __name__ == '__main__':
    min_threshold = 1e9
    max_threshold = -1e9
    best_recipe = None
    best_books = None
    worst_recipe = None
    thresFunctoUse = BEC_findThresForRecipe
    menus = [(1,1,1,1),(1,1,1,0),(1,1,0,1),(1,0,1,1),(0,1,1,1)]#,(1,1,0,0),(1,0,1,0),(1,0,0,1),(0,1,1,0),(0,1,0,1),(0,0,1,1)]
    blockSize = 317;L = 6;truncate = 1

    for composition in find_composition(L,len(menus)):
        recipe = {k:v for k,v in zip(menus,composition)}
        if not isRecipeSatisfyRect(recipe): continue
        thres = thresFunctoUse(recipe,blockSize,truncate)
        if thres > max_threshold:
            max_threshold = thres
            worst_recipe = recipe
            #best_books = var_book, chk_book
        if thres < min_threshold:
            min_threshold = thres
            best_recipe = recipe
    if best_recipe is None or worst_recipe is None:
        raise ValueError('No recipe found')
    best_recipe = {k:v for k,v in best_recipe.items() if v > 0}
    worst_recipe = {k:v for k,v in worst_recipe.items() if v > 0}

    print(f'Best recipe is {best_recipe}\n\t Threshold: {min_threshold} dB')
    print(f'Worst recipe is {worst_recipe}\n\t Threshold: {max_threshold} dB')
