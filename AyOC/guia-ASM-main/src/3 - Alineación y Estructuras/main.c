#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>

#include "../test-utils.h"
#include "Estructuras.h"

int main() {
	/* Acá pueden realizar sus propias pruebas */
	nodo_t *head = malloc(sizeof(nodo_t));
	if(head == NULL){
		free(head);
		return 1;
	}
	uint32_t *arr = malloc(3 * sizeof(uint32_t));
	if(head == NULL){
		free(head);
		return 1;
	}
	head->next = NULL;
	head->arreglo = arr;
	head->longitud = 3;
	head->categoria = 3;

	arr[0] = 1;
	arr[1] = 2;
	arr[2] = 3;

	lista_t *lista;
	lista->head = head;

	printf("%u\n", cantidad_total_de_elementos(lista));

	free(head);
	free(arr);

	return 0;
}
