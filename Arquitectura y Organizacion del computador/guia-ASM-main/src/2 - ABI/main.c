#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>

#include "../test-utils.h"
#include "ABI.h"

int main() {
	/* Acá pueden realizar sus propias pruebas */

	assert(alternate_sum_8(822, 230, 481, 566, 592, 70, 838, 216) == 1651);

	// int32_t res;
	// int32_t* p = &res;
	// int32_t x1 = 2;
	// float x2 = 2.0;
	// product_2_f(p, x1, x2);
	// assert(*p == 4);
	// return 0;

	double res;
	double *p = &res;
	int32_t x1 = 2;
	float f1 = 2;
	product_9_f(p, 4, 4.0, 2, 2.0, 2, 2.0, 2, 2.0, 2, 2.0, 2, 2.0, 2, 2.0, 2, 2.0, 2, 2.0);
	printf("el resultado es: %f \n", *p);
