import heat as heat
import fem as fem


if __name__ == "__main__":
    # heat.example_problem()
    print(fem.gl2_quadrature_integration(lambda x: x[0]**2, [0], [1], 1))
    print(fem.gl2_quadrature_integration(lambda x: x[0]**2 + x[1]**2, [0, 0], [1, 1], 2))

    value = fem.gl2_quadrature_integration(
        lambda x: x[0]**2 + x[1]**2 + x[2]**2,
        [0, 0, 0],
        [1, 1, 1],
        3
    )

    print(value)

