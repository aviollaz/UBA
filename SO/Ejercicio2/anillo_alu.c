#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>
#include "constants.h"

enum {READ, WRITE};
enum {STDIN, STDOUT};

int generate_random_number(){
	return (rand() % 50);
}

int main(int argc, char **argv)
{	
	//Funcion para cargar nueva semilla para el numero aleatorio
	srand(time(NULL));

	int status, pid, n, start, buffer;
	n = atoi(argv[1]);
	buffer = atoi(argv[2]);
	start = atoi(argv[3]);

	pid_t *children = malloc(sizeof(*children) * n);

	if (argc != 4){ printf("Uso: anillo <n> <c> <s> \n"); exit(0);}
    
    printf("Se crearán %i procesos, se enviará el caracter %i desde proceso %i \n", n, buffer, start);
    
	int pipes_hijos[n+2][2];

	for (int i = 0; i < n + 2; i++){
		if (pipe(pipes_hijos[i]) == -1){
			exit(1);
		}
	}
	for (int i = 0; i < n; i++){
		pid_t pid = fork();
		if(pid == 0){
			//Cada hijo cierra los pipes del anillo que no usa
			for (int j = 0; j < n; j++){
				if (j != (i+1)%n){
					close(pipes_hijos[j][READ]);
				}		
				if (j != i){
					close(pipes_hijos[j][WRITE]);
				}
			}

			//los procesos no elegidos cierran el pipe con el padre
			//y el elegido cierra las 2 entradas que no usa
			if (i != start){
				close(pipes_hijos[n][READ]);
				close(pipes_hijos[n][WRITE]);
				close(pipes_hijos[n+1][READ]);
				close(pipes_hijos[n+1][WRITE]);
			}
			else{
				close(pipes_hijos[n][WRITE]);
				close(pipes_hijos[n+1][READ]);
			}
			
			int numero_inicial;

			if(i == start){
				
				int numero_secreto;
				read(pipes_hijos[n][READ], &numero_inicial, sizeof(numero_inicial));
				numero_secreto = generate_random_number();
				printf("El numero secreto es: %d\n", numero_secreto);
				while(numero_inicial < numero_secreto){
					write(pipes_hijos[i][WRITE], &numero_inicial, sizeof(numero_inicial));
					read(pipes_hijos[(i+1)%n][READ], &numero_inicial, sizeof(numero_inicial));
					numero_inicial++;
					printf("soy: %d, el numero que tengo es: %d\n", i, numero_inicial);
				}

				write(pipes_hijos[n+1][WRITE], &numero_inicial, sizeof(numero_inicial));
				numero_inicial = -1;
				write(pipes_hijos[i][WRITE], &numero_inicial, sizeof(numero_inicial));
			}
			else{	
				read(pipes_hijos[(i+1)%n][READ], &numero_inicial, sizeof(numero_inicial));

				while(numero_inicial != -1){
					numero_inicial++;
					printf("soy: %d, el numero que tengo es: %d\n", i, numero_inicial);
					write(pipes_hijos[i][WRITE], &numero_inicial, sizeof(numero_inicial));
					read(pipes_hijos[(i+1)%n][READ], &numero_inicial, sizeof(numero_inicial));
				}

				write(pipes_hijos[i][WRITE], &numero_inicial, sizeof(numero_inicial));
			}
			exit(0);
		}
		children[i] = pid;
	}

	//El padre cierra los pipes del anillo
	for (int j = 0; j < n; j++){
		close(pipes_hijos[j][WRITE]);
		close(pipes_hijos[j][READ]);		
	}
	//cierra los write y read que no usa
	close(pipes_hijos[n+1][WRITE]);
	close(pipes_hijos[n][READ]);

	write(pipes_hijos[n][WRITE], &buffer, sizeof(buffer));
	int numero_final;
	read(pipes_hijos[n+1][READ], &numero_final, sizeof(numero_final));

	
	//Espero a los hijos y verifico el estado que terminaron
	for (int i = 0; i < n; i++) {
		waitpid(children[i], &status, 0);

		if (!WIFEXITED(status)) {
			printf("el procesp que fallo es: %d\n", i);	//-----a veces da error o algo asi, preguntar
			fprintf(stderr, "proceso %d no terminó correctamente [%d]: ",
			    (int)children[i], WIFSIGNALED(status));
			perror("");
			return -1;
		}
	}
	free(children);
		
	printf("El numero magico es: %d\n", numero_final);  
	return 0;
}
