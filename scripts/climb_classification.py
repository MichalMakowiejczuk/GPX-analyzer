

def classify_climb_difficulty(dl_km, gradient):
    """ Calculates the difficulty score of a climb based on its length and gradient (just like strava: https://support.strava.com/hc/en-us/articles/216917057-Climb-Categorization).
    Args:
        dl_km (float): Length of the climb in meters.
        gradient (float): Average gradient of the climb in percent.
    Returns: 
        tuple: A tuple containing the category as a string and the score as a float.
    """
    score = dl_km* (gradient)
    
    if score < 8000:
        category = "Very easy (Cat <4)"
    elif score < 16000: # 8 - 16
        category = "Easy (Cat 4)"
    elif score < 32000:
        category = "Medium (Cat 3)"
    elif score < 64000:
        category = "Hard (Cat 2)"
    elif score < 80000:
        category = "Very hard (Cat 1)"
    else:
        category = "Hors catégorie (HC)"
    
    return category, score

if __name__ == "__main__":
    # Example usage
    # Length in km and gradient in percent
    # For example, a climb of 2.3 km with an average gradient of 4.1%
    length = 2.3  # in m
    gradient = 4.1  # in percent

    category, score = classify_ascent_difficulty(length, gradient)
    print(f"Category: {category}, Score: {score:.2f}")