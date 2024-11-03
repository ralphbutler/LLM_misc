import random


def generate_random_integers(count=5, start=1, end=100):
    """
    Generate a list of random integers.

    :param count: Number of random integers to generate.
    :param start: The lower bound of the random integers (inclusive).
    :param end: The upper bound of the random integers (inclusive).
    :return: A list of random integers.
    """
    random_integers = []
    for _ in range(count):
        # Generate a random integer between start and end
        random_integer = random.randint(start, end)
        random_integers.append(random_integer)
    return random_integers


def main():
    """
    Main function to execute the script.

    Generates five random integers and prints them.
    """
    # Generate five random integers
    random_integers = generate_random_integers()
    
    # Print the generated random integers
    print("Generated random integers:", random_integers)


if __name__ == "__main__":
    main()
