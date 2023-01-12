#ifndef EDMONDS_SERIAL_H
#define EDMONDS_SERIAL_H

#include "CSRGraphRep.h"
#include <bits/stdc++.h>
using namespace std;

/*

https://codeforces.com/blog/entry/49402

GETS:
V->number of vertices
E->number of edges
pair of vertices as edges (vertices are 1..V)

GIVES:
output of edmonds() is the maximum matching
match[i] is matched pair of i (-1 if there isn't a matched pair)
 */
struct struct_edge{int v;struct_edge* n;};
typedef struct_edge* edge;

 
class EdmondsSerial {
  public:
    EdmondsSerial(CSRGraph & _G) : 
                    G(_G),
                    V(G.vertexNum)
    {
      match = (int *)malloc(V * sizeof(int));
      q = (int *)malloc(V * sizeof(int));
      father = (int *)malloc(V * sizeof(int));
      base = (int *)malloc(V * sizeof(int));

      inq = (bool *)malloc(V * sizeof(bool));
      inb = (bool *)malloc(V * sizeof(bool));
      
    }
    ~EdmondsSerial(){
      free(match);
      free(q);
      free(father);
      free(base);
      free(inq);
      free(inb);
    }

    int* get_match(){return match;}

    int edmonds();

    private:
        CSRGraph &G;

        int V,*match,qh,qt,*q,*father,*base;
        bool *inq,*inb;

        void add_edge(int u,int v);
        int LCA(int root,int u,int v);
        void mark_blossom(int lca,int u);
        void blossom_contraction(int s,int u,int v);
        int find_augmenting_path(int s);
        int augment_path(int s,int t);

};
 
int EdmondsSerial::LCA(int root,int u,int v)
{
  static bool *inp;
  inp = (bool *)malloc(V * sizeof(bool)); 
  memset(inp,0,V*sizeof(bool));
  while(1)
    {
      inp[u=base[u]]=true;
      if (u==root) break;
      u=father[match[u]];
    }
  while(1)
    {
      if (inp[v=base[v]]) return v;
      else v=father[match[v]];
    }
    free(inp);
}
 
void EdmondsSerial::mark_blossom(int lca,int u)
{
  while (base[u]!=lca)
    {
      int v=match[u];
      inb[base[u]]=inb[base[v]]=true;
      u=father[v];
      if (base[u]!=lca) father[u]=v;
    }
}
 
void EdmondsSerial::blossom_contraction(int s,int u,int v)
{
  int lca=LCA(s,u,v);
  memset(inb,0,V*sizeof(bool));
  mark_blossom(lca,u);
  mark_blossom(lca,v);
  if (base[u]!=lca)
    father[u]=v;
  if (base[v]!=lca)
    father[v]=u;
  for (int u=0;u<V;u++)
    if (inb[base[u]])
      {
	base[u]=lca;
	if (!inq[u])
	  inq[q[++qt]=u]=true;
      }
}
 
int EdmondsSerial::find_augmenting_path(int s)
{
  memset(inq,0,V*sizeof(bool));
  memset(father,-1,V*sizeof(int));
  for (int i=0;i<V;i++) base[i]=i;
  inq[q[qh=qt=0]=s]=true;
  while (qh<=qt)
    {
      int u=q[qh++];
      // adj list is 2d array of edges.
      // edges store v and a ptr to the next e in the array.
      // this avoids a counter for each row.
      // the final edge has a null ptr hence
      // e is false and the loop terminates.
      // therefore, this should iterate over all the 
      // edges belonging to u.
      // it's possible this could be parallelized in the future.

      // adj list
      // for (edge e=adj[u];e;e=e->n)
    for (unsigned int j = G.srcPtr[u]; j < G.srcPtr[u + 1]; ++j){
          unsigned int v =G.dst[j];
          //int v=e->v;
          if (base[u]!=base[v]&&match[u]!=v){
            if ((v==s)||(match[v]!=-1 && father[match[v]]!=-1)){
              blossom_contraction(s,u,v);
            } else if (father[v]==-1)
            {
              father[v]=u;
              if (match[v]==-1)
                return v;
              else if (!inq[match[v]])
                inq[q[++qt]=match[v]]=true;
            }
          }
        }
    }
  return -1;
}
 
int EdmondsSerial::augment_path(int s,int t)
{
  int u=t,v,w;
  while (u!=-1)
    {
      v=father[u];
      w=match[v];
      match[v]=u;
      match[u]=v;
      u=w;
    }
  return t!=-1;
}
 
int EdmondsSerial::edmonds()
{
  int matchc=0;
  memset(match,-1,V*sizeof(int));
  for (int u=0;u<V;u++)
    if (match[u]==-1)
      matchc+=augment_path(u,find_augmenting_path(u));
  /*
  for (int i=0;i<V;i++)
    if (i<match[i])
      cout<<i<<" "<<match[i]<<endl;
  */
  return matchc;
}
#endif