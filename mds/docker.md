#Docker tips

## Remove the <none> images
```
docker rmi -f $(docker images | grep "<none>" | awk "{print \$3}")
```

