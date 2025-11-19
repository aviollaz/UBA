#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>

int n;
int numero_maldito;
pid_t PID_hijos[10];

int generate_random_number(){
    //Funcion para cargar nueva semilla para el numero aleatorio
    srand(time(NULL) ^ getpid());
    return (rand() % n);
}

void signal_17_handler(int sig){
    pid_t pid = wait(NULL);
    
    // guardar hijos suicidados
    for(int i = 0; i < n; i++){
        if(PID_hijos[i] == pid){
            PID_hijos[i] = -1;
        }
    }
    return;
}

void signal_15_handler(int sig){
    int num = generate_random_number();

    if(num == numero_maldito){
        printf("sopas, mi pid es %d\n", getpid());
        pid_t parent_pid = getppid();
        kill(parent_pid, 17);
        exit(0);    // VER
    }
    return;
}

int main(int argc, char const *argv[]){
    
    n = atoi(argv[1]);
	int rondas = atoi(argv[2]);
	numero_maldito = atoi(argv[3]);

    signal(17, signal_17_handler);

    // creo y guardo PIDs     
    pid_t pid;
    
    for(int i = 0; i < n; i++){
        pid = fork();

        if(pid == 0){
            signal(15, signal_15_handler);
            printf("%d", i);
            printf("\n while de %d\n", getpid());
            while(1);

        } else{
            PID_hijos[i] = pid;
        }
    }


    if(pid != 0){
        sleep(2);
        for(int i = 0; i < rondas; i++){
            for(int j = 0; j < n; j++){
                if(PID_hijos[j] != -1){
                    kill(PID_hijos[j], 15); // en cada ronda envio SIGTERM a cada hijo
                }
                sleep(1);
            }
        }

        for(int i = 0; i < n; i++){
            if(PID_hijos[i] != -1){
                kill(PID_hijos[i], 9);  
                pid_t pid_hijo = wait(NULL);
                printf("estoy vivo, mi PID es: %d\n", pid_hijo);
            }
        }

    }
    exit(0);
}
