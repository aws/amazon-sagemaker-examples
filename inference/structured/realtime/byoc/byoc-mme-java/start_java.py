from subprocess import check_output

result = check_output(["java", "-Djava.security.egd=file:/dev/./urandom", "-Dapp.port=${app.port}", "-jar","/work/app.jar"])