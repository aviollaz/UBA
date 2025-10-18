#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>
#include "constants.h"
#include "mini-shell-parser.c"

enum {READ, WRITE};
enum {STDIN, STDOUT};

static int run(char ***progs, size_t count)
{	
	//crear pipes
	int pipes_hijos[count - 1][2];

	for (int i = 0; i < count - 1; i++){
		int no_me_uses = pipe(pipes_hijos[i]);
	}
	
	int r, status;

	pid_t *children = malloc(sizeof(*children) * count);

	for (int i = 0; i < count; i++){
		pid_t pid = fork();
		if(pid == 0){

			for (int j = 0; j < count - 1; j++){
					//si no es tu pipe ni el del anterior
				if (j != i && j != i-1){
					close(pipes_hijos[j][WRITE]);
					close(pipes_hijos[j][READ]);
				}	//si es tu pipe
				else if (j == i){
					close(pipes_hijos[j][READ]);
				}	//si es el pipe del anterior
				else if (j == i-1){
					close(pipes_hijos[j][WRITE]);
				}		
			}
			
			if (i == 0){
				dup2(pipes_hijos[i][WRITE], STDOUT);
				//como el primer programa (hijo) ya tiene toda la informacion que necesita para ejecutarce
				//lo ejecutamos
				execvp(progs[i][0], progs[i]);
			}
			else if (i != count-1){
				dup2(pipes_hijos[i][WRITE], STDOUT);
				dup2(pipes_hijos[i-1][READ], STDIN);
				execvp(progs[i][0], progs[i]);
			}
			else{
				dup2(pipes_hijos[i-1][READ], STDIN);
				execvp(progs[i][0], progs[i]);
			}

			exit(0);
		}
		
		children[i] = pid;
	}

	//El padre cierra todo
	for (int j = 0; j < count - 1; j++){
		close(pipes_hijos[j][WRITE]);
		close(pipes_hijos[j][READ]);		
	}
	
	//Espero a los hijos y verifico el estado que terminaron
	for (int i = 0; i < count; i++) {
		waitpid(children[i], &status, 0);

		if (!WIFEXITED(status)) {
			fprintf(stderr, "proceso %d no terminó correctamente [%d]: ",
			    (int)children[i], WIFSIGNALED(status));
			perror("");
			return -1;
		}
	}
	r = 0;
	free(children);
	return r;
}


int main(int argc, char **argv)
{
	if (argc < 2) {
		printf("El programa recibe como parametro de entrada un string con la linea de comandos a ejecutar. \n"); 
		printf("Por ejemplo ./mini-shell 'ls -a | grep anillo'\n");
		return 0;
	}
	int programs_count;
	char*** programs_with_parameters = parse_input(argv, &programs_count);
	//^ puntero a array de strings
	printf("status: %d\n", run(programs_with_parameters, programs_count));

	fflush(stdout);
	fflush(stderr);
	return 0;
}
