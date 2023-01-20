#ifndef EDMONDS_PARALELL_H
#define EDMONDS_PARALELL_H

#include "CSRGraphRep.h"
#include <bits/stdc++.h>
using namespace std;
#include <CL/sycl.hpp>

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

class EdmondsParallel {
  public:
    EdmondsParallel(CSRGraph & _G,
                    sycl::buffer<int> &_match,
                    sycl::buffer<int> &_q,
                    sycl::buffer<int> &_father,
                    sycl::buffer<int> &_base,
                    sycl::buffer<bool> &_inq,
                    sycl::buffer<bool> &_inb) : 
                    G(_G),
                    V(G.vertexNum),
                    match(_match),   
                    q(_q),   
                    father(_father),
                    base(_base),
                    inq(_inq),                    
                    inb(_inb) {

      //inb = sycl::buffer<bool>{VertexSize};

      /*
      match = (int *)malloc(V * sizeof(int));
      q = (int *)malloc(V * sizeof(int));
      father = (int *)malloc(V * sizeof(int));
      base = (int *)malloc(V * sizeof(int));

      inq = (bool *)malloc(V * sizeof(bool));
      inb = (bool *)malloc(V * sizeof(bool));
      */
      
    }
    ~EdmondsParallel(){

    }


    int edmonds();

    private:
        CSRGraph &G;
        // Device output vector
        sycl::buffer<int> &match,&q,&father,&base;
        int V,qh,qt;
        sycl::buffer<bool> &inq,&inb;

        //const sycl::range RowSize;
        //const sycl::range ColSize;
        //const sycl::range VertexSize;

        /*
        void add_edge(int u,int v);
        int LCA(int root,int u,int v);
        void mark_blossom(int lca,int u);
        void blossom_contraction(int s,int u,int v);
        int find_augmenting_path(int s);
        int augment_path(int s,int t);
        */
};
/*
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

// Unmatched vertices find paths (which may be of length 0) to augment.
int EdmondsSerial::find_augmenting_path(int s)
{
  memset(inq,0,V*sizeof(bool));
  memset(father,-1,V*sizeof(int));
  for (int i=0;i<V;i++) base[i]=i;
  // q (paths sourced at s) length n
  // inq (bool array for easy check if in a path src'ed at s)
  inq[q[qh=qt=0]=s]=true;
  while (qh<=qt)
    {
      // Here is where q is initialized (the path is extended)
      // This is a post-fix so on first call q[0] = s; and qh == 1;
      // u == s
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
          // At the begginning of this method call base[i] == i
          // Every vertex is a blossom of size 1 based at itself.
          // though since inb[i]==false for all i, no blossom logic is used.
          // u and v are in different blossoms and u isnt matched to v.
          if (base[u]!=base[v]&&match[u]!=v){
            // Since u and v arent matched, v==s indicates an alternate edge into the src
            // v==s indicates blossoms rooted at s.
            // If v is matched with a father, the dfs must have formed a cycle.
            // If father[match[v]]!=-1) then the path is at least length 3. x-f[v]-matchedto-v
            if ((v==s)||(match[v]!=-1 && father[match[v]]!=-1)){
              blossom_contraction(s,u,v);


            // Grow ontop of the tree, the path is lengthen by finding src's and telling them
            // I am your father.  If src is unmatched, return an augmented path length 2.
            // else if the vertex hasnt been encountered on this call to findAP, mark it as reached,
            // and continue the search from the match of v. i.e. src = u; v-match[v]
            // iter 0: u->v-match[v]; set u=match[v]
            // iter 1: u-notmatched-v-matched-match[v]
            } else if (father[v]==-1)
            {
              // grow dfs depth by 1. set pred v -> u
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
 */

int EdmondsParallel::edmonds()
{
  int matchc=0;
  //memset(match,-1,V*sizeof(int));
  //for (int u=0;u<V;u++)
    //if (match[u]==-1)
      //matchc+=augment_path(u,find_augmenting_path(u));

  return matchc;
}
#endif